"""FastAPI service wrapping the retrieval engine.

The DB is opened read-only once; engines (FAISS index + in-RAM codes) are loaded
lazily per collection and cached for the process lifetime -- the "semi-persistent"
service the GUI / agent talk to.
"""
import os
import io
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from .store import PatchStore
from .engine import RetrievalEngine

_LOCK = threading.Lock()
_STATE = {"store": None, "engines": {}}


def _store():
    if _STATE["store"] is None:
        db = os.environ.get("PATCHDB_DB", "runs/patchdb/patchdb.duckdb")
        _STATE["store"] = PatchStore(db, read_only=True)
    return _STATE["store"]


def _engine(name):
    with _LOCK:
        if name not in _STATE["engines"]:
            try:
                _STATE["engines"][name] = RetrievalEngine(_store(), name)
            except KeyError as e:
                raise HTTPException(404, str(e))
            except RuntimeError as e:
                raise HTTPException(409, str(e))
        return _STATE["engines"][name]


@asynccontextmanager
async def lifespan(app):
    try:
        store = _store()
        for c in store.list_collections():           # warm-load every collection
            try:
                _engine(c["name"])
            except HTTPException:
                pass
        app.state.ready = True
    except Exception as e:                            # noqa: BLE001 - serve anyway
        app.state.ready = False
        app.state.error = str(e)
    yield


app = FastAPI(title="patchdb", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    collection: str
    bbox: List[int]                                  # [row, col, h, w]
    image_id: Optional[int] = None
    dataset_index: Optional[int] = None
    px: bool = False
    topk: int = 12
    whiten: bool = True
    min_fg_frac: float = 0.6
    topc: int = 400


class VectorQueryRequest(BaseModel):
    collection: str
    vec: List[float]
    h: int
    w: int
    topk: int = 12
    whiten: bool = True
    min_fg_frac: float = 0.6
    topc: int = 400


@app.get("/healthz")
def healthz():
    return {"status": "ok", "ready": getattr(app.state, "ready", False)}


@app.get("/collections")
def collections():
    return {"collections": _store().list_collections()}


@app.get("/collections/{name}")
def collection_info(name: str):
    store = _store()
    try:
        cid = store.get_collection_id(name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    m = store.get_model(cid)
    return {
        "collection": name, "id": cid, "k": m["k"], "grid": m["grid"],
        "patch_size": m["patch_size"], "img_size": m["img_size"],
        "whiten_index": m["whiten_index"], "fg_thresh": m["fg_thresh"],
        "top_ev": (m["ev"][:5] * 100).round(2).tolist(),
        "faiss": store.get_faiss(cid),
    }


@app.post("/query")
def query(req: QueryRequest):
    eng = _engine(req.collection)
    if len(req.bbox) != 4:
        raise HTTPException(422, "bbox must be [row, col, h, w]")
    r, c, h, w = req.bbox
    if req.px:
        r, c, h, w = r // eng.ps, c // eng.ps, max(1, h // eng.ps), max(1, w // eng.ps)
    if req.image_id is None and req.dataset_index is None:
        raise HTTPException(422, "provide image_id or dataset_index")
    try:
        res = eng.query(image_id=req.image_id, dataset_index=req.dataset_index,
                        bbox=(r, c, h, w), topk=req.topk, whiten=req.whiten,
                        min_fg_frac=req.min_fg_frac, topc=req.topc)
    except KeyError as e:
        raise HTTPException(404, str(e))
    qid = req.image_id
    if qid is None:
        qid = eng.store.image_meta(eng.cid, dataset_index=req.dataset_index)["id"]
    return {"collection": req.collection, "query_image_id": qid,
            "bbox_patch": [r, c, h, w], "whiten": req.whiten, "results": res}


@app.post("/query/by-vector")
def query_by_vector(req: VectorQueryRequest):
    eng = _engine(req.collection)
    try:
        res = eng.query_vector(vec=req.vec, h=req.h, w=req.w, topk=req.topk,
                               whiten=req.whiten, min_fg_frac=req.min_fg_frac,
                               topc=req.topc)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"collection": req.collection, "h": req.h, "w": req.w,
            "whiten": req.whiten, "results": res}


@app.get("/image/{name}/{image_id}.png")
def image_png(name: str, image_id: int):
    eng = _engine(name)
    with _LOCK:
        try:
            img = eng.load_image(image_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.imsave(buf, np.asarray(img), cmap="gray", format="png")
    return Response(content=buf.getvalue(), media_type="image/png")
