"""Build / extend / reindex a collection: encode images, persist, index.

Separated from query (`engine.py`) so the heavy torch path is only imported when
constructing data. ``reindex_collection`` is pure-numpy (no net) and rebuilds the
FAISS index + token map from the stored code grids.
"""
import os

import numpy as np
import torch

from .encoder import (resolve_ckpt, ckpt_epoch_tag, load_net, make_dataset,
                      get_view, extract_tokens, fit_shared_basis, project_view)
from .pool import whiten_scale, normalize_codes
from . import index as faiss_index


def _foreground_vectors(codes, fg, ev, whiten):
    """Flatten foreground tokens of one image -> (vectors[M,K], gi[M], gj[M])."""
    ii, jj = np.nonzero(fg)
    vecs = codes[ii, jj]                              # [M, K] raw projections
    vecs = normalize_codes(vecs, whiten_scale(ev, whiten))
    return vecs.astype(np.float32), ii.astype(np.int32), jj.astype(np.int32)


def reindex_collection(store, collection_id, *, faiss_path=None,
                       index_type="flat"):
    """Rebuild the FAISS index + token map (numpy sidecar) from stored codes.

    The token map is an int32 ``[ntotal, 3]`` array of ``(image ord, gi, gj)``
    aligned to FAISS vector ids (vectors are added in ord order).
    """
    model = store.get_model(collection_id)
    G, K, ev = model["grid"], model["k"], model["ev"]
    whiten = model["whiten_index"]
    codes, fg, image_ids, _ = store.load_codes_array(collection_id, G, K)
    N = codes.shape[0]

    all_vecs, tok_n, tok_gi, tok_gj = [], [], [], []
    for n in range(N):
        vecs, gi, gj = _foreground_vectors(codes[n], fg[n], ev, whiten)
        all_vecs.append(vecs)
        tok_n.append(np.full(vecs.shape[0], n, dtype=np.int32))
        tok_gi.append(gi)
        tok_gj.append(gj)
    vectors = np.concatenate(all_vecs, 0) if all_vecs else np.zeros((0, K), np.float32)
    token_map = (np.stack([np.concatenate(tok_n), np.concatenate(tok_gi),
                           np.concatenate(tok_gj)], axis=1)
                 if all_vecs else np.zeros((0, 3), np.int32)).astype(np.int32)

    base = os.path.dirname(os.path.abspath(store.db_path))
    if faiss_path is None:
        faiss_path = os.path.join(base, f"faiss_c{collection_id}.index")
    token_path = os.path.splitext(faiss_path)[0] + ".tokens.npy"

    idx = faiss_index.build_index(vectors, index_type=index_type)
    faiss_index.write_index(idx, faiss_path)
    np.save(token_path, token_map)
    store.set_faiss(collection_id, path=faiss_path, token_path=token_path,
                    ntotal=int(idx.ntotal), dim=int(K), metric="ip",
                    index_type=index_type)
    return {"ntotal": int(idx.ntotal), "path": faiss_path,
            "token_path": token_path, "n_images": N}


def _encode_images(store, collection_id, net, ds, enc_ids, mu, Vh, device, *,
                   pattern, backend, data_dir, dataset_key, fg_thresh,
                   start_ord, progress=True):
    """Encode + store a set of dataset indices; returns number stored."""
    for r, di in enumerate(enc_ids):
        codes, fg = project_view(net, get_view(ds, di), mu, Vh, device,
                                 fg_thresh=fg_thresh)
        store.add_image(
            collection_id, ord=start_ord + r, dataset_index=di,
            source_uri=f"{pattern}#{di}", pattern=pattern, backend=backend,
            data_dir=data_dir, dataset_key=dataset_key, n_fg=int(fg.sum()),
            codes=codes, fg=fg)
        if progress and ((r + 1) % 100 == 0 or r + 1 == len(enc_ids)):
            print(f"  encoded {r + 1}/{len(enc_ids)}", flush=True)
    return len(enc_ids)


def build_collection(store, *, name, run_dir, ckpt_subdir="ckpt",
                     eigen_ckpt="last", data_dir=".", pattern="soild_stack.zarr",
                     backend="zarr", dataset_key="reconstruction", img_size=512,
                     in_chans=1, proj_dim=16, k=25, n_fit=64, n_images=0,
                     seed=0, foreground_mask=True, fg_std_thresh=0.05,
                     outlier_pct=2.0, whiten_index=True, index_type="flat",
                     faiss_path=None, device=None, replace=True):
    """Encode a stack onto a freshly fit shared basis, persist + index it."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fg_thresh = fg_std_thresh if foreground_mask else None
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt = resolve_ckpt(run_dir, ckpt_subdir, eigen_ckpt)
    net = load_net(ckpt, img_size=img_size, in_chans=in_chans,
                   proj_dim=proj_dim, device=device)
    ds = make_dataset(data_dir, pattern, backend, dataset_key, img_size)
    n_total = len(ds)
    n_enc = n_total if n_images <= 0 else min(n_images, n_total)
    enc_ids = list(range(n_enc))

    # fit shared basis on a sample
    rng = np.random.default_rng(seed)
    fit_ids = sorted(rng.choice(n_enc, size=min(n_fit, n_enc),
                                replace=False).tolist())
    fg_pool = []
    with torch.no_grad():
        for i in fit_ids:
            tokens, grid, fg = extract_tokens(net, get_view(ds, i), device,
                                              fg_thresh=fg_thresh)
            fg_pool.append(tokens[fg])
    mu, Vh, ev = fit_shared_basis(torch.cat(fg_pool, 0), k,
                                  outlier_pct=outlier_pct)
    K, D = Vh.shape[0], Vh.shape[1]
    G = grid
    ps = img_size // G
    print(f"fit basis on {len(fit_ids)} imgs  K={K}  top-EV={ev[:3] * 100}",
          flush=True)

    cid = store.create_collection(name, params={
        "run_dir": run_dir, "ckpt": ckpt, "pattern": pattern,
        "fg_std_thresh": fg_std_thresh, "whiten_index": whiten_index,
    }, replace=replace)
    store.set_model(
        cid, ckpt_path=ckpt, k=K, embed_dim=D, grid=G, patch_size=ps,
        img_size=img_size, whiten_index=whiten_index, outlier_pct=outlier_pct,
        fg_thresh=fg_thresh, basis=Vh.cpu().numpy(), mean=mu.cpu().numpy(),
        ev=ev)

    with torch.no_grad():
        _encode_images(store, cid, net, ds, enc_ids, mu, Vh, device,
                       pattern=pattern, backend=backend, data_dir=data_dir,
                       dataset_key=dataset_key, fg_thresh=fg_thresh, start_ord=0)

    info = reindex_collection(store, cid, faiss_path=faiss_path,
                              index_type=index_type)
    info.update({"collection": name, "collection_id": cid, "k": K, "grid": G,
                 "patch_size": ps, "epoch": ckpt_epoch_tag(ckpt)})
    return info


def add_to_collection(store, *, name, data_dir=None, pattern=None, backend=None,
                      dataset_key=None, n_images=0, proj_dim=16, in_chans=1,
                      index_type="flat", device=None):
    """Encode another stack onto an existing collection's basis, then reindex."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cid = store.get_collection_id(name)
    model = store.get_model(cid)
    fg_thresh = model["fg_thresh"]
    img_size = model["img_size"]
    # reuse the existing source metadata unless overridden
    sample = store.con.execute(
        "SELECT pattern, backend, data_dir, dataset_key FROM images "
        "WHERE collection_id = ? LIMIT 1", [cid]).fetchone()
    pattern = pattern or sample[0]
    backend = backend or sample[1]
    data_dir = data_dir or sample[2]
    dataset_key = dataset_key or sample[3]

    net = load_net(model["ckpt_path"], img_size=img_size, in_chans=in_chans,
                   proj_dim=proj_dim, device=device)
    mu = torch.from_numpy(model["mean"]).to(device)
    Vh = torch.from_numpy(model["basis"]).to(device)
    ds = make_dataset(data_dir, pattern, backend, dataset_key, img_size)
    n_total = len(ds)
    n_enc = n_total if n_images <= 0 else min(n_images, n_total)
    start_ord = store.next_image_ord(cid)
    with torch.no_grad():
        _encode_images(store, cid, net, ds, list(range(n_enc)), mu, Vh, device,
                       pattern=pattern, backend=backend, data_dir=data_dir,
                       dataset_key=dataset_key, fg_thresh=fg_thresh,
                       start_ord=start_ord)
    info = reindex_collection(store, cid, index_type=index_type)
    info.update({"collection": name, "collection_id": cid, "added": n_enc})
    return info
