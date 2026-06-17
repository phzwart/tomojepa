"""RetrievalEngine: load a collection into RAM and run hybrid queries.

Hybrid = FAISS candidate generation over per-token vectors, then an exact
integral-image re-rank at the query's true window size, restricted to the images
that produced candidates. This keeps results exact for any window size while
scaling sub-linearly in the corpus.
"""
import numpy as np

from .store import PatchStore
from .pool import (whiten_scale, normalize_codes, integral_window_mean,
                   window_mean_single)
from . import index as faiss_index


class RetrievalEngine:
    def __init__(self, store, collection_name):
        self.store = store
        self.name = collection_name
        self.cid = store.get_collection_id(collection_name)
        self.model = store.get_model(self.cid)
        self.G = self.model["grid"]
        self.K = self.model["k"]
        self.ps = self.model["patch_size"]
        self.ev = self.model["ev"]

        # in-RAM code grids (source of truth = DuckDB)
        self.codes, self.fg, self.image_ids, self.dataset_idx = \
            store.load_codes_array(self.cid, self.G, self.K)
        self.N = self.codes.shape[0]
        max_id = int(self.image_ids.max()) if self.N else 0
        self._id2n = np.full(max_id + 1, -1, dtype=np.int64)
        self._id2n[self.image_ids] = np.arange(self.N)

        # FAISS index + token map (numpy sidecar: [ntotal, 3] = ord, gi, gj)
        fmeta = store.get_faiss(self.cid)
        if fmeta is None:
            raise RuntimeError(f"collection '{collection_name}' has no FAISS index")
        self.index = faiss_index.load_index(fmeta["path"])
        token_map = np.load(fmeta["token_path"])
        self._tok_n = token_map[:, 0].astype(np.int64)
        self._tok_gi = token_map[:, 1].astype(np.int64)
        self._tok_gj = token_map[:, 2].astype(np.int64)

        self._index_scale = whiten_scale(self.ev, self.model["whiten_index"])
        self._lazy_encoder = None
        self._ds_cache = {}

    # ---- helpers ---------------------------------------------------------
    def n_of_image(self, image_id):
        if 0 <= image_id < len(self._id2n) and self._id2n[image_id] >= 0:
            return int(self._id2n[image_id])
        raise KeyError(f"image_id {image_id} not in collection '{self.name}'")

    def _clamp_bbox(self, r, c, h, w):
        h = max(1, min(int(h), self.G))
        w = max(1, min(int(w), self.G))
        r = max(0, min(int(r), self.G - h))
        c = max(0, min(int(c), self.G - w))
        return r, c, h, w

    def _encoder(self):
        if self._lazy_encoder is None:
            import torch
            from .encoder import load_net, SharedBasisEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net = load_net(self.model["ckpt_path"], img_size=self.model["img_size"],
                           in_chans=1, proj_dim=16, device=device)
            mu = torch.from_numpy(self.model["mean"]).to(device)
            Vh = torch.from_numpy(self.model["basis"]).to(device)
            self._lazy_encoder = SharedBasisEncoder(
                net, mu, Vh, self.ev, self.G, self.ps, self.model["img_size"],
                self.model["fg_thresh"], device)
        return self._lazy_encoder

    def _dataset(self, pattern, backend, data_dir, dataset_key):
        key = (pattern, backend, data_dir, dataset_key)
        if key not in self._ds_cache:
            from .encoder import make_dataset
            self._ds_cache[key] = make_dataset(data_dir, pattern, backend,
                                               dataset_key, self.model["img_size"])
        return self._ds_cache[key]

    def load_image(self, image_id):
        """Grayscale image [H, W] for a stored image_id (for viz / GUI)."""
        meta = self.store.image_meta(self.cid, image_id=image_id)
        if meta is None:
            raise KeyError(f"image_id {image_id} not found")
        from .encoder import get_view
        ds = self._dataset(meta["pattern"], meta["backend"], meta["data_dir"],
                           meta["dataset_key"])
        view = get_view(ds, meta["dataset_index"])
        return view[0].mean(0).cpu().numpy()

    # ---- query -----------------------------------------------------------
    def _rank(self, qraw, cand_ns, h, w, *, whiten, topk, min_fg_frac,
              exclude=None):
        """Exact re-rank of windows of size h x w over candidate images.

        ``qraw``: raw query projection [K]. ``cand_ns``: unique candidate image
        positions. ``exclude``: optional (n, r, c) window to drop.
        """
        scale = whiten_scale(self.ev, whiten)
        qn = normalize_codes(qraw, scale)
        cand_ns = np.asarray(sorted(set(int(n) for n in cand_ns)), dtype=np.int64)
        desc, frac = integral_window_mean(self.codes[cand_ns], self.fg[cand_ns],
                                          h, w)                  # [M, Y, X, K]
        M, Y, X, _ = desc.shape
        dn = normalize_codes(desc.reshape(-1, self.K), scale).reshape(M, Y, X, self.K)
        sim = dn @ qn
        sim = np.where(frac >= min_fg_frac, sim, -2.0)
        if exclude is not None:
            en, er, ec = exclude
            wpos = np.where(cand_ns == en)[0]
            if len(wpos) and er < Y and ec < X:
                sim[int(wpos[0]), er, ec] = -2.0

        # greedy spatial NMS within each image
        order = np.argsort(sim.reshape(-1))[::-1]
        picks, taken = [], {}
        for idx in order:
            s = float(sim.reshape(-1)[idx])
            if s <= -1.0:
                break
            mi, y, x = np.unravel_index(idx, sim.shape)
            mi, y, x = int(mi), int(y), int(x)
            ok = True
            for (yy, xx) in taken.get(mi, []):
                if abs(yy - y) < h and abs(xx - x) < w:
                    ok = False
                    break
            if not ok:
                continue
            taken.setdefault(mi, []).append((y, x))
            n = int(cand_ns[mi])
            picks.append((s, n, y, x))
            if len(picks) >= topk:
                break
        return picks

    def _format(self, picks, h, w):
        out = []
        for rank, (s, n, y, x) in enumerate(picks):
            out.append({
                "rank": rank + 1,
                "similarity": round(float(s), 4),
                "image_id": int(self.image_ids[n]),
                "dataset_index": int(self.dataset_idx[n]),
                "patch": {"row": y, "col": x, "h": h, "w": w},
                "bbox_px": [y * self.ps, x * self.ps, h * self.ps, w * self.ps],
            })
        return out

    def query(self, *, image_id=None, dataset_index=None, bbox, topk=12,
              whiten=True, min_fg_frac=0.6, topc=400):
        """Retrieve similar windows for a bbox in an image already in the DB.

        ``bbox``: (row, col, h, w) in PATCH units. Returns a list of match dicts.
        """
        if image_id is None and dataset_index is not None:
            image_id = self.store.image_meta(
                self.cid, dataset_index=dataset_index)["id"]
        n = self.n_of_image(image_id)
        r, c, h, w = self._clamp_bbox(*bbox)
        qraw = window_mean_single(self.codes[n], self.fg[n], r, c, h, w)
        return self._run(qraw, h, w, topk=topk, whiten=whiten,
                         min_fg_frac=min_fg_frac, topc=topc, exclude=(n, r, c))

    def query_external(self, *, view, bbox, topk=12, whiten=True,
                       min_fg_frac=0.6, topc=400):
        """Retrieve for a bbox in an image NOT in the DB (encoded on the fly).

        ``view``: tensor [1, C, H, W]. ``bbox``: (row, col, h, w) patch units.
        """
        codes, fg = self._encoder().encode_view(view)
        r, c, h, w = self._clamp_bbox(*bbox)
        qraw = window_mean_single(codes, fg, r, c, h, w)
        return self._run(qraw, h, w, topk=topk, whiten=whiten,
                         min_fg_frac=min_fg_frac, topc=topc)

    def query_vector(self, *, vec, h, w, topk=12, whiten=True, min_fg_frac=0.6,
                     topc=400):
        """Retrieve given a raw K-dim code vector and a target window size."""
        qraw = np.asarray(vec, dtype=np.float32).reshape(-1)
        if qraw.shape[0] != self.K:
            raise ValueError(f"vector dim {qraw.shape[0]} != K={self.K}")
        h = max(1, min(int(h), self.G)); w = max(1, min(int(w), self.G))
        return self._run(qraw, h, w, topk=topk, whiten=whiten,
                         min_fg_frac=min_fg_frac, topc=topc)

    def _run(self, qraw, h, w, *, topk, whiten, min_fg_frac, topc, exclude=None):
        qvec = normalize_codes(qraw, self._index_scale)[None]
        _, ids = faiss_index.search(self.index, qvec, topc)
        ids = ids[0]
        ids = ids[ids >= 0]
        if ids.size == 0:
            return []
        cand_ns = self._tok_n[ids]
        picks = self._rank(qraw, cand_ns, h, w, whiten=whiten, topk=topk,
                           min_fg_frac=min_fg_frac, exclude=exclude)
        return self._format(picks, h, w)


def open_engine(db_path, collection_name):
    return RetrievalEngine(PatchStore(db_path), collection_name)
