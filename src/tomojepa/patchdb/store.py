"""DuckDB-backed structured store for collections, models, images, and codes.

BLOBs hold raw numpy bytes; shapes are reconstructed from the model row. Bulk
tensors (code grids) live here as the source of truth and are materialized into
RAM by the engine at load time. The FAISS index and its vectors live in sidecar
files referenced by the ``faiss_index`` table.
"""
import os
import json

import numpy as np
import duckdb

_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def _b(arr, dtype):
    return np.ascontiguousarray(arr, dtype=dtype).tobytes()


class PatchStore:
    def __init__(self, db_path, read_only=False):
        self.db_path = db_path
        self.read_only = read_only
        if read_only:
            self.con = duckdb.connect(db_path, read_only=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".",
                        exist_ok=True)
            self.con = duckdb.connect(db_path)
            self._init_schema()

    def _init_schema(self):
        with open(_SCHEMA) as f:
            self.con.execute(f.read())

    def close(self):
        # Flush the WAL into the main DB file so a later open never has to replay
        # it (avoids a DuckDB WAL-replay assertion after an unclean prior exit).
        if not self.read_only:
            try:
                self.con.execute("CHECKPOINT")
            except Exception:
                pass
        self.con.close()

    # ---- write -----------------------------------------------------------
    def create_collection(self, name, params=None, replace=False):
        existing = self.con.execute(
            "SELECT id FROM collections WHERE name = ?", [name]).fetchone()
        if existing is not None:
            if not replace:
                return int(existing[0])
            self.drop_collection(name)
        self.con.execute(
            "INSERT INTO collections (name, params) VALUES (?, ?)",
            [name, json.dumps(params or {})])
        return int(self.con.execute(
            "SELECT id FROM collections WHERE name = ?", [name]).fetchone()[0])

    def drop_collection(self, name):
        row = self.con.execute(
            "SELECT id FROM collections WHERE name = ?", [name]).fetchone()
        if row is None:
            return
        cid = int(row[0])
        self.con.execute(
            "DELETE FROM codes WHERE image_id IN "
            "(SELECT id FROM images WHERE collection_id = ?)", [cid])
        for t in ("images", "models", "faiss_index"):
            self.con.execute(f"DELETE FROM {t} WHERE collection_id = ?", [cid])
        self.con.execute("DELETE FROM collections WHERE id = ?", [cid])

    def set_model(self, collection_id, *, ckpt_path, k, embed_dim, grid,
                  patch_size, img_size, whiten_index, outlier_pct, fg_thresh,
                  basis, mean, ev):
        self.con.execute("DELETE FROM models WHERE collection_id = ?",
                         [collection_id])
        self.con.execute(
            "INSERT INTO models (collection_id, ckpt_path, k, embed_dim, grid, "
            "patch_size, img_size, whiten_index, outlier_pct, fg_thresh, basis, "
            "mean, ev) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [collection_id, ckpt_path, int(k), int(embed_dim), int(grid),
             int(patch_size), int(img_size), bool(whiten_index),
             float(outlier_pct), (None if fg_thresh is None else float(fg_thresh)),
             _b(basis, np.float32), _b(mean, np.float32), _b(ev, np.float32)])

    def next_image_ord(self, collection_id):
        row = self.con.execute(
            "SELECT COALESCE(MAX(ord) + 1, 0) FROM images WHERE collection_id = ?",
            [collection_id]).fetchone()
        return int(row[0])

    def add_image(self, collection_id, *, ord, dataset_index, source_uri,
                  pattern, backend, data_dir, dataset_key, n_fg, codes, fg):
        self.con.execute(
            "INSERT INTO images (collection_id, ord, dataset_index, source_uri, "
            "pattern, backend, data_dir, dataset_key, n_fg) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [collection_id, int(ord), int(dataset_index), source_uri, pattern,
             backend, data_dir, dataset_key, int(n_fg)])
        image_id = int(self.con.execute(
            "SELECT id FROM images WHERE collection_id = ? AND ord = ?",
            [collection_id, int(ord)]).fetchone()[0])
        self.con.execute(
            "INSERT INTO codes (image_id, codes, fg) VALUES (?,?,?)",
            [image_id, _b(codes, np.float16), _b(fg, np.bool_)])
        return image_id

    def set_faiss(self, collection_id, *, path, token_path, ntotal, dim, metric,
                  index_type):
        self.con.execute("DELETE FROM faiss_index WHERE collection_id = ?",
                         [collection_id])
        self.con.execute(
            "INSERT INTO faiss_index (collection_id, path, token_path, ntotal, "
            "dim, metric, index_type) VALUES (?,?,?,?,?,?,?)",
            [collection_id, path, token_path, int(ntotal), int(dim), metric,
             index_type])

    # ---- read ------------------------------------------------------------
    def list_collections(self):
        rows = self.con.execute(
            "SELECT c.name, c.id, c.created_at, "
            "(SELECT COUNT(*) FROM images i WHERE i.collection_id = c.id), "
            "(SELECT COALESCE(MAX(f.ntotal), 0) FROM faiss_index f "
            "WHERE f.collection_id = c.id) "
            "FROM collections c ORDER BY c.name").fetchall()
        return [{"name": r[0], "id": int(r[1]), "created_at": str(r[2]),
                 "n_images": int(r[3]), "n_tokens": int(r[4])} for r in rows]

    def get_collection_id(self, name):
        row = self.con.execute(
            "SELECT id FROM collections WHERE name = ?", [name]).fetchone()
        if row is None:
            raise KeyError(f"collection '{name}' not found")
        return int(row[0])

    def get_model(self, collection_id):
        r = self.con.execute(
            "SELECT ckpt_path, k, embed_dim, grid, patch_size, img_size, "
            "whiten_index, outlier_pct, fg_thresh, basis, mean, ev "
            "FROM models WHERE collection_id = ?", [collection_id]).fetchone()
        if r is None:
            raise KeyError(f"no model for collection {collection_id}")
        k, D = int(r[1]), int(r[2])
        return {
            "ckpt_path": r[0], "k": k, "embed_dim": D, "grid": int(r[3]),
            "patch_size": int(r[4]), "img_size": int(r[5]),
            "whiten_index": bool(r[6]), "outlier_pct": float(r[7]),
            "fg_thresh": (None if r[8] is None else float(r[8])),
            "basis": np.frombuffer(r[9], dtype=np.float32).reshape(k, D).copy(),
            "mean": np.frombuffer(r[10], dtype=np.float32).reshape(1, D).copy(),
            "ev": np.frombuffer(r[11], dtype=np.float32).reshape(k).copy(),
        }

    def get_faiss(self, collection_id):
        r = self.con.execute(
            "SELECT path, token_path, ntotal, dim, metric, index_type "
            "FROM faiss_index WHERE collection_id = ?", [collection_id]).fetchone()
        if r is None:
            return None
        return {"path": r[0], "token_path": r[1], "ntotal": int(r[2]),
                "dim": int(r[3]), "metric": r[4], "index_type": r[5]}

    def load_codes_array(self, collection_id, grid, k):
        """Materialize all code grids + fg + ids ordered by ``ord``.

        Returns ``(codes[N,G,G,k] f32, fg[N,G,G] bool, image_ids[N], dataset_idx[N])``.
        """
        rows = self.con.execute(
            "SELECT i.ord, i.id, i.dataset_index, cd.codes, cd.fg "
            "FROM images i JOIN codes cd ON cd.image_id = i.id "
            "WHERE i.collection_id = ? ORDER BY i.ord", [collection_id]).fetchall()
        N = len(rows)
        codes = np.zeros((N, grid, grid, k), dtype=np.float32)
        fg = np.zeros((N, grid, grid), dtype=bool)
        image_ids = np.zeros(N, dtype=np.int64)
        dataset_idx = np.zeros(N, dtype=np.int64)
        for n, (ordv, iid, didx, cb, fb) in enumerate(rows):
            codes[n] = np.frombuffer(cb, dtype=np.float16).reshape(grid, grid, k)
            fg[n] = np.frombuffer(fb, dtype=np.bool_).reshape(grid, grid)
            image_ids[n] = int(iid)
            dataset_idx[n] = int(didx)
        return codes, fg, image_ids, dataset_idx

    def image_meta(self, collection_id, image_id=None, dataset_index=None):
        if image_id is not None:
            q = ("SELECT id, dataset_index, pattern, backend, data_dir, "
                 "dataset_key, source_uri FROM images WHERE collection_id = ? "
                 "AND id = ?")
            args = [collection_id, int(image_id)]
        else:
            q = ("SELECT id, dataset_index, pattern, backend, data_dir, "
                 "dataset_key, source_uri FROM images WHERE collection_id = ? "
                 "AND dataset_index = ? ORDER BY id LIMIT 1")
            args = [collection_id, int(dataset_index)]
        r = self.con.execute(q, args).fetchone()
        if r is None:
            return None
        return {"id": int(r[0]), "dataset_index": int(r[1]), "pattern": r[2],
                "backend": r[3], "data_dir": r[4], "dataset_key": r[5],
                "source_uri": r[6]}
