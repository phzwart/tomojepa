"""FAISS index build / persist / search.

Vectors are expected pre-whitened + L2-normalized, so inner-product search ==
cosine similarity. ``IndexFlatIP`` is exact and sub-millisecond at this scale;
IVF / HNSW are available for larger corpora.
"""
import numpy as np
import faiss


def build_index(vectors, index_type="flat", nlist=None, hnsw_m=32):
    """Build a FAISS inner-product index over ``vectors`` [N, K] (float32).

    ``index_type``: 'flat' (exact), 'ivf' (IVFFlat), or 'hnsw' (HNSWFlat).
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = vectors.shape
    it = index_type.lower()
    if it == "flat":
        index = faiss.IndexFlatIP(d)
    elif it == "ivf":
        nlist = nlist or max(1, min(int(4 * np.sqrt(n)), max(1, n // 39)))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.nprobe = min(nlist, 16)
    elif it == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"unknown index_type {index_type!r}")
    index.add(vectors)
    return index


def write_index(index, path):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    faiss.write_index(index, path)


def load_index(path):
    return faiss.read_index(path)


def search(index, queries, topc):
    """Return ``(scores[Q, topc], ids[Q, topc])`` for inner-product search."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None]
    return index.search(queries, int(topc))
