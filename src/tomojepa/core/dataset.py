import os
import glob
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import (build_augmentations, build_probe_augmentation, build_fg_mask,
                            wrap_image_mask, unwrap_image_mask, DynamicAugStore)
from .aug_config import AugmentationConfig, AugmentationSchedule


def _resolve_array(root, key):
    """Return the indexable array from an opened volume.

    Handles both container objects (h5py ``File`` / zarr ``Group``, which hold
    the array under ``key``) and stores that are themselves the array
    (zarr ``Array`` written without a parent group).
    """
    if hasattr(root, "shape"):              # already an array / dataset (no `key`)
        return root
    # Container (h5py File / zarr Group): look the array up by key. We check
    # `.shape` first because `key in <array>` does an element-wise value test.
    if key in root:
        return root[key]
    raise KeyError(key)


class TomographyDataset(Dataset):
    """Lazily slices a directory of microCT volumes into 2D views.

    Supports HDF5 (``.h5``) and Zarr (``.zarr``) backends, auto-detected per
    file from the extension (override with ``backend``). Each volume holds a
    ``(D, H, W)`` array under ``dataset_key``; a flat global index is mapped to
    ``(file, depth-slice)``. Handles are opened lazily *inside* ``__getitem__``
    and LRU-capped, which keeps this safe with ``num_workers > 0`` (each worker
    opens its own handles after fork).
    """

    def __init__(self, data_dir, dataset_key="reconstruction", pattern="recon_*.h5",
                 global_views=2, local_views=2,
                 global_scale=(0.4, 1.0), local_scale=(0.1, 0.4),
                 variant="tomo2", img_size=512, is_train=True, max_open_files=64,
                 backend="auto", crop_mode="resized", crop_size=None,
                 foreground_mask=False, fg_mode="std", fg_std_thresh=0.05,
                 fg_circle_diameter_frac=1.0, fg_key=None,
                 random_rotate_deg=None, probe_geom=False,
                 resize_jitter_scale=None,
                 aug_config: AugmentationConfig | None = None,
                 aug_schedule: AugmentationSchedule | None = None,
                 shared_aug_state: bool = False):
        self.data_dir = data_dir
        self.dataset_key = dataset_key
        self.fg_key = fg_key
        self.global_views = global_views
        self.local_views = local_views
        self.V = global_views + local_views
        self.is_train = is_train
        self.max_open_files = max_open_files
        self.foreground_mask = foreground_mask
        self.fg_mode = fg_mode
        self.fg_std_thresh = fg_std_thresh
        self.fg_circle_diameter_frac = fg_circle_diameter_frac
        if backend not in ("auto", "h5", "zarr"):
            raise ValueError(f"unknown backend: {backend!r}")
        self.backend = backend

        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")

        self.scan_infos = []
        self.total_len = 0
        for fpath in self.files:
            container, arr = self._open_volume(fpath)
            try:
                if arr is None:
                    continue
                shape = tuple(arr.shape)
                if len(shape) < 3:                               # need at least (D, H, W)
                    continue
                # Accept (D, H, W) or (D, C, H, W); depth is first, spatial is last two.
                d, h, w = shape[0], shape[-2], shape[-1]
                self.scan_infos.append(
                    {"path": fpath, "start": self.total_len,
                     "end": self.total_len + d, "shape": (d, h, w)}
                )
                self.total_len += d
            finally:
                self._close(container)

        if self.total_len == 0:
            raise ValueError(
                f"Found {len(self.files)} file(s) but none contained dataset "
                f"key {dataset_key!r}."
            )

        self.random_rotate_deg = random_rotate_deg
        self.resize_jitter_scale = resize_jitter_scale
        self.img_size = img_size
        self.crop_size = crop_size
        self.probe_geom = probe_geom

        if aug_config is None:
            from .augmentations import _legacy_config_from_kwargs
            aug_config = _legacy_config_from_kwargs(
                variant, crop_mode, global_scale, local_scale,
                random_rotate_deg, resize_jitter_scale)
        else:
            aug_config = aug_config
        if global_views != aug_config.global_views or local_views != aug_config.local_views:
            from dataclasses import replace
            aug_config = replace(aug_config,
                                 global_views=global_views, local_views=local_views)
        self.aug_config = aug_config
        self.aug_schedule = aug_schedule
        self.global_views = aug_config.global_views
        self.local_views = aug_config.local_views
        self.V = self.global_views + self.local_views
        self.random_rotate_deg = aug_config.random_rotate_deg
        self.resize_jitter_scale = aug_config.resize_jitter_scale()

        use_dynamic = aug_schedule is not None and is_train and not probe_geom
        self._aug_dynamic = DynamicAugStore()
        self._aug_dynamic.update_from_config(aug_config)
        if use_dynamic and shared_aug_state:
            self._aug_dynamic.enable_shared()

        self._rebuild_transforms()
        self._open = OrderedDict()

    def _rebuild_transforms(self):
        cfg = self.aug_config
        dynamic = self._aug_dynamic if (
            self.aug_schedule is not None and self.is_train and not self.probe_geom
        ) else None
        if self.probe_geom:
            self.test_tf = build_probe_augmentation(
                cfg, self.img_size, self.foreground_mask, crop_size=self.crop_size)
            self.global_tf = self.local_tf = self.test_tf
        else:
            self.global_tf, self.local_tf, self.test_tf = build_augmentations(
                cfg, self.img_size, self.foreground_mask, self.crop_size, dynamic)

    def update_augmentations(self, step: int, total_steps: int,
                             steps_per_epoch: int | None = None) -> None:
        """Apply scheduled augmentation overrides at ``step`` (no-op if static)."""
        if self.aug_schedule is None or not self.is_train or self.probe_geom:
            return
        state = self.aug_schedule.at(
            self.aug_config, step, total_steps, steps_per_epoch)
        self._aug_dynamic.update_from_state(state)
        self.random_rotate_deg = state.random_rotate_deg
        self.resize_jitter_scale = state.resize_jitter

    def __len__(self):
        return self.total_len

    def _backend_for(self, path):
        if self.backend != "auto":
            return self.backend
        p = path.lower().rstrip("/")
        if p.endswith(".zarr") or p.endswith(".zarr.zip"):
            return "zarr"
        return "h5"

    def _open_volume(self, path):
        """Open a volume, returning ``(container, array_or_None)``.

        ``container`` is what must be closed on eviction; ``array_or_None`` is
        ``None`` when the file exists but lacks ``dataset_key``.
        """
        if self._backend_for(path) == "zarr":
            import zarr
            root = zarr.open(path, mode="r")
        else:
            import h5py
            root = h5py.File(path, "r")
        try:
            return root, _resolve_array(root, self.dataset_key)
        except KeyError:
            return root, None

    @staticmethod
    def _close(container):
        close = getattr(container, "close", None)   # h5py File; zarr is a no-op
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _locate(self, idx):
        for info in self.scan_infos:
            if info["start"] <= idx < info["end"]:
                return info, idx - info["start"]
        raise IndexError(idx)

    def _array(self, path):
        entry = self._open.get(path)
        if entry is not None:                                    # mark most-recently-used
            self._open.move_to_end(path)
            return entry[1]
        if len(self._open) >= self.max_open_files:               # evict least-recently-used
            self._close(self._open.popitem(last=False)[1][0])
        container, arr = self._open_volume(path)
        self._open[path] = (container, arr)
        return arr

    def _load_fg_mask(self, path, local, img: torch.Tensor) -> torch.Tensor:
        """Return ``[1, H, W]`` float foreground mask for slice ``local``."""
        if self.fg_key is not None:
            container, root = self._open_volume(path)
            try:
                fg_arr = _resolve_array(root, self.fg_key)
                fg = np.ascontiguousarray(fg_arr[local], dtype=np.float32)
                if fg.ndim == 3:
                    fg = fg[0]
                fg_t = torch.from_numpy(fg)
                if fg_t.dim() == 2:
                    fg_t = fg_t.unsqueeze(0)
                return (fg_t > 0.5).float()
            except KeyError:
                pass
            finally:
                self._close(container)
        return build_fg_mask(img, self.fg_mode, self.fg_std_thresh,
                             self.fg_circle_diameter_frac)

    def _apply_tf(self, tf, img, fg=None):
        if self.foreground_mask:
            if fg is None:
                fg = build_fg_mask(img, self.fg_mode, self.fg_std_thresh,
                                   self.fg_circle_diameter_frac)
            sample = wrap_image_mask(img, fg)
            img, fg = unwrap_image_mask(tf(sample))
            return img, fg
        return tf(img), None

    def __getitem__(self, idx):
        info, local = self._locate(idx)
        try:
            arr = np.ascontiguousarray(self._array(info["path"])[local], dtype=np.float32)
            if arr.ndim == 2:                                    # (H, W) -> (1, H, W)
                img = torch.from_numpy(arr)[None]
            else:                                                # (C, H, W), already channel-first
                img = torch.from_numpy(arr)
        except Exception:                                        # corrupt slice -> skip
            _, h, w = info["shape"]
            img = torch.zeros((1, h, w), dtype=torch.float32)

        fg_src = (self._load_fg_mask(info["path"], local, img)
                  if self.foreground_mask else None)

        if self.is_train:
            views, fg_views = [], []
            for _ in range(self.global_views):
                v, f = self._apply_tf(self.global_tf, img, fg_src)
                views.append(v)
                fg_views.append(f)
            for _ in range(self.local_views):
                v, f = self._apply_tf(self.local_tf, img, fg_src)
                views.append(v)
                fg_views.append(f)
            if self.foreground_mask:
                return torch.stack(views), torch.stack(fg_views)
            return torch.stack(views), 0
        view, fg = self._apply_tf(self.test_tf, img, fg_src)
        if self.foreground_mask:
            return view, fg
        return view, 0


# Backwards-compatible alias (the class used to be HDF5-only).
TomographyH5Dataset = TomographyDataset
