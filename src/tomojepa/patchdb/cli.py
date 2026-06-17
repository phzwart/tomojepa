"""patchdb command-line interface (Typer).

Commands:
  build    encode a stack onto a fresh shared basis -> DuckDB + FAISS
  add      append another stack to an existing collection (reuse basis)
  reindex  rebuild the FAISS index + token map from stored codes
  info     list collections / show one collection's model + counts
  query    retrieve similar patches (JSON to stdout and/or a PNG)
  serve    launch the FastAPI service
  mcp      launch the MCP stdio server (thin client to the service)
"""
import os
import json

import typer

app = typer.Typer(add_completion=False, help="FAISS patch-retrieval engine.")

DEFAULT_DB = os.environ.get("PATCHDB_DB", "runs/patchdb/patchdb.duckdb")


def _parse_bbox(bbox: str):
    parts = [int(x) for x in bbox.replace(",", " ").split()]
    if len(parts) != 4:
        raise typer.BadParameter("bbox must be 'ROW COL H W'")
    return parts


@app.command()
def build(
    collection: str = typer.Option(..., help="collection name"),
    run_dir: str = typer.Option(..., help="run dir with ckpt/"),
    db: str = typer.Option(DEFAULT_DB),
    eigen_ckpt: str = typer.Option("last", help="'last', epoch int, or a path"),
    data_dir: str = typer.Option("."),
    pattern: str = typer.Option("soild_stack.zarr"),
    backend: str = typer.Option("zarr"),
    dataset_key: str = typer.Option("reconstruction"),
    img_size: int = typer.Option(512),
    k: int = typer.Option(25, help="components stored per token"),
    n_fit: int = typer.Option(64, help="images sampled to fit the basis"),
    n_images: int = typer.Option(0, help="images to encode (0 = all)"),
    foreground_mask: bool = typer.Option(True),
    fg_std_thresh: float = typer.Option(0.05),
    outlier_pct: float = typer.Option(2.0),
    whiten_index: bool = typer.Option(True),
    index_type: str = typer.Option("flat", help="flat | ivf | hnsw"),
    replace: bool = typer.Option(True, help="replace collection if it exists"),
):
    """Build a new collection."""
    from .store import PatchStore
    from .builder import build_collection
    store = PatchStore(db)
    info = build_collection(
        store, name=collection, run_dir=run_dir, eigen_ckpt=eigen_ckpt,
        data_dir=data_dir, pattern=pattern, backend=backend,
        dataset_key=dataset_key, img_size=img_size, k=k, n_fit=n_fit,
        n_images=n_images, foreground_mask=foreground_mask,
        fg_std_thresh=fg_std_thresh, outlier_pct=outlier_pct,
        whiten_index=whiten_index, index_type=index_type, replace=replace)
    store.close()
    typer.echo(json.dumps(info, indent=2, default=str))


@app.command()
def add(
    collection: str = typer.Option(...),
    db: str = typer.Option(DEFAULT_DB),
    pattern: str = typer.Option(None, help="defaults to collection's source"),
    data_dir: str = typer.Option(None),
    backend: str = typer.Option(None),
    dataset_key: str = typer.Option(None),
    n_images: int = typer.Option(0),
    index_type: str = typer.Option("flat"),
):
    """Append another stack to an existing collection."""
    from .store import PatchStore
    from .builder import add_to_collection
    store = PatchStore(db)
    info = add_to_collection(
        store, name=collection, pattern=pattern, data_dir=data_dir,
        backend=backend, dataset_key=dataset_key, n_images=n_images,
        index_type=index_type)
    store.close()
    typer.echo(json.dumps(info, indent=2, default=str))


@app.command()
def reindex(
    collection: str = typer.Option(...),
    db: str = typer.Option(DEFAULT_DB),
    index_type: str = typer.Option("flat"),
):
    """Rebuild the FAISS index + token map from stored codes."""
    from .store import PatchStore
    from .builder import reindex_collection
    store = PatchStore(db)
    cid = store.get_collection_id(collection)
    info = reindex_collection(store, cid, index_type=index_type)
    store.close()
    typer.echo(json.dumps(info, indent=2, default=str))


@app.command()
def info(
    db: str = typer.Option(DEFAULT_DB),
    collection: str = typer.Option(None, help="show details for one collection"),
):
    """List collections, or show one collection's model + counts."""
    from .store import PatchStore
    store = PatchStore(db)
    if collection is None:
        out = {"db": db, "collections": store.list_collections()}
    else:
        cid = store.get_collection_id(collection)
        m = store.get_model(cid)
        fmeta = store.get_faiss(cid)
        out = {
            "collection": collection, "id": cid,
            "k": m["k"], "grid": m["grid"], "patch_size": m["patch_size"],
            "img_size": m["img_size"], "embed_dim": m["embed_dim"],
            "whiten_index": m["whiten_index"], "fg_thresh": m["fg_thresh"],
            "ckpt": m["ckpt_path"], "top_ev": (m["ev"][:5] * 100).round(2).tolist(),
            "faiss": fmeta,
            "n_images": store.list_collections() and next(
                (c["n_images"] for c in store.list_collections()
                 if c["name"] == collection), None),
        }
    store.close()
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command()
def query(
    collection: str = typer.Option(...),
    bbox: str = typer.Option(..., help="'ROW COL H W'"),
    db: str = typer.Option(DEFAULT_DB),
    image: int = typer.Option(None, help="query image_id"),
    dataset_index: int = typer.Option(None, help="query by source slice index"),
    px: bool = typer.Option(False, help="interpret bbox in pixels"),
    topk: int = typer.Option(12),
    whiten: bool = typer.Option(True),
    min_fg_frac: float = typer.Option(0.6),
    topc: int = typer.Option(400, help="FAISS candidates before re-rank"),
    out: str = typer.Option(None, help="render matches to this PNG"),
    json_out: bool = typer.Option(True, "--json/--no-json"),
):
    """Retrieve similar patches for a query window."""
    from .store import PatchStore
    from .engine import RetrievalEngine
    store = PatchStore(db)
    eng = RetrievalEngine(store, collection)
    r, c, h, w = _parse_bbox(bbox)
    if px:
        r, c, h, w = r // eng.ps, c // eng.ps, max(1, h // eng.ps), max(1, w // eng.ps)
    if image is None and dataset_index is None:
        raise typer.BadParameter("provide --image or --dataset-index")
    res = eng.query(image_id=image, dataset_index=dataset_index,
                    bbox=(r, c, h, w), topk=topk, whiten=whiten,
                    min_fg_frac=min_fg_frac, topc=topc)
    qid = image if image is not None else store.image_meta(
        eng.cid, dataset_index=dataset_index)["id"]
    if out:
        from .viz import render_query
        render_query(eng, res, bbox=(r, c, h, w), query_image_id=qid, out_path=out)
        typer.echo(f"wrote {out}")
    if json_out:
        typer.echo(json.dumps(
            {"collection": collection, "query_image_id": qid,
             "bbox_patch": [r, c, h, w], "whiten": whiten, "results": res},
            indent=2, default=str))
    store.close()


@app.command()
def serve(
    db: str = typer.Option(DEFAULT_DB),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8077),
):
    """Launch the FastAPI service."""
    os.environ["PATCHDB_DB"] = db
    import uvicorn
    uvicorn.run("tomojepa.patchdb.service:app", host=host, port=port, reload=False)


@app.command()
def mcp(
    service_url: str = typer.Option("http://127.0.0.1:8077",
                                    envvar="PATCHDB_SERVICE_URL"),
):
    """Launch the MCP stdio server (proxies to the FastAPI service)."""
    os.environ["PATCHDB_SERVICE_URL"] = service_url
    from .mcp_server import run
    run()


if __name__ == "__main__":
    app()
