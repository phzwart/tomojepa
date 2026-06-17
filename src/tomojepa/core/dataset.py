import os
import glob
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import get_augmentations


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
                 backend="auto"):
        self.data_dir = data_dir
        self.dataset_key = dataset_key
        self.global_views = global_views
        self.local_views = local_views
        self.V = global_views + local_views
        self.is_train = is_train
        self.max_open_files = max_open_files
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

        self.global_tf, self.local_tf, self.test_tf = get_augmentations(
            variant=variant, img_size=img_size,
            global_scale=global_scale, local_scale=local_scale,
        )
        self._open = OrderedDict()

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

        if self.is_train:
            views = ([self.global_tf(img) for _ in range(self.global_views)]
                     + [self.local_tf(img) for _ in range(self.local_views)])
            return torch.stack(views), 0
        return self.test_tf(img), 0


# Backwards-compatible alias (the class used to be HDF5-only).
TomographyH5Dataset = TomographyDataset
