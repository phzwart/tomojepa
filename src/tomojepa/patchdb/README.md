# patchdb — FAISS patch-retrieval engine

Encodes images into shared-basis per-token codes, indexes the foreground tokens
in FAISS, and retrieves similar regions of **any window size** across an image
database. Backed by DuckDB for structured metadata and exposed via a CLI, a
FastAPI service, and an MCP tool wrapper.

## How it works

1. **Encode** — load a trained encoder (`runs/<run>/ckpt/ckpt_epoch_*.pth`), fit
   one shared PCA basis on foreground patch tokens pooled across a sample of
   images, and project every image's tokens onto it -> a `G x G x K` code grid
   (default 32x32x25) plus a foreground mask. Because the basis is shared, codes
   are comparable across images.
2. **Index** — flatten the foreground tokens (whitened + L2-normalized) into a
   FAISS inner-product index (cosine). A numpy sidecar maps each vector id back
   to `(image, gi, gj)`.
3. **Query (hybrid)** — summarize a query window as the foreground-weighted mean
   of its codes, use FAISS to fetch candidate token centers fast, then do an
   exact integral-image re-rank at the query's true window size over just the
   candidate images. Exact for any size; scales sub-linearly in the corpus.

`--whiten` (default on) rescales each component by `1/sqrt(eigenvalue)` so fine
texture, not just overall density (PC1 ~ porosity, ~97% variance), drives the
match.

## Storage layout

A collection lives under the DuckDB file's directory:

- `patchdb.duckdb` — collections, the shared-basis model (basis/mean/ev as
  blobs), per-image metadata, and per-image code grids (source of truth).
- `faiss_c<id>.index` — the FAISS index.
- `faiss_c<id>.tokens.npy` — int32 `[ntotal, 3]` token map `(image ord, gi, gj)`.

Geometry: patch `(i, j)` <-> pixels `[i*ps:(i+1)*ps, j*ps:(j+1)*ps]` (`ps` =
`patch_size`, stored on the model).

## CLI

Run from the repo root with the project venv (`.venv/bin/python -m patchdb.cli ...`).

```bash
# Build a collection from a trained run + a zarr stack
python -m patchdb.cli build --collection soil \
    --run-dir runs/soil_residual_fg --eigen-ckpt 14 \
    --pattern soild_stack.zarr --k 25 --n-fit 64

# Inspect
python -m patchdb.cli info                       # list collections
python -m patchdb.cli info --collection soil     # model + index details

# Query (bbox = "ROW COL H W" in patch units; add --px for pixels)
python -m patchdb.cli query --collection soil \
    --dataset-index 484 --bbox "14 14 5 5" --topk 12 \
    --out runs/patchdb/q484.png                  # JSON to stdout + PNG

# Append another stack onto the same basis, or rebuild the index
python -m patchdb.cli add --collection soil --pattern other_stack.zarr
python -m patchdb.cli reindex --collection soil --index-type flat
```

`--image` selects by internal `image_id`; `--dataset-index` selects by the
source slice index in the stack.

## Service (FastAPI)

```bash
PATCHDB_DB=runs/patchdb/patchdb.duckdb python -m patchdb.cli serve --port 8077
```

Endpoints: `GET /healthz`, `GET /collections`, `GET /collections/{name}`,
`POST /query`, `POST /query/by-vector`, `GET /image/{name}/{image_id}.png`.
The DB is opened read-only and every collection is warm-loaded once
(FAISS index + code grids held in RAM) for low-latency reuse.

```bash
curl -s -X POST localhost:8077/query -H 'Content-Type: application/json' \
  -d '{"collection":"soil","dataset_index":484,"bbox":[14,14,5,5],"topk":12}'
```

## MCP (agent tool)

Start the service, then run the MCP stdio server (it proxies to the service):

```bash
python -m patchdb.cli serve --port 8077        # terminal 1
python -m patchdb.cli mcp                       # terminal 2 (PATCHDB_SERVICE_URL)
```

`.cursor/mcp.json` registers it for the Cursor agent. Tools: `list_collections`,
`collection_info`, `patch_search`, `get_image_url`.

## Notes

- Index types: `flat` (exact, default; sub-ms search at ~1M vectors), `ivf`,
  `hnsw` for larger corpora.
- The prototype scripts `build_token_db.py` / `query_patches.py` are superseded
  by this package.
