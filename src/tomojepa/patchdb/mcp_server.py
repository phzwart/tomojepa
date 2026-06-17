"""MCP stdio server exposing patch retrieval as agent tools.

This is a thin client over the FastAPI service (``PATCHDB_SERVICE_URL``) so the
heavy FAISS index / code arrays live in exactly one process. Start the service
first (``patchdb serve``), then run ``patchdb mcp``.
"""
import os
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

SERVICE_URL = os.environ.get("PATCHDB_SERVICE_URL", "http://127.0.0.1:8077")

mcp = FastMCP("patchdb")


def _client():
    return httpx.Client(base_url=SERVICE_URL, timeout=30.0)


@mcp.tool()
def list_collections() -> dict:
    """List available patch-retrieval collections and their sizes."""
    with _client() as c:
        return c.get("/collections").json()


@mcp.tool()
def collection_info(name: str) -> dict:
    """Show a collection's geometry (grid, patch size), basis info, and index."""
    with _client() as c:
        r = c.get(f"/collections/{name}")
        if r.status_code != 200:
            return {"error": r.text, "status": r.status_code}
        return r.json()


@mcp.tool()
def patch_search(
    collection: str,
    row: int,
    col: int,
    h: int,
    w: int,
    image_id: Optional[int] = None,
    dataset_index: Optional[int] = None,
    px: bool = False,
    topk: int = 12,
    whiten: bool = True,
) -> dict:
    """Find regions similar to a query window across the image database.

    The query window is (row, col, h, w). By default these are PATCH units; set
    ``px=True`` to give pixels. Identify the query image by ``image_id`` or by
    its source ``dataset_index``. ``whiten`` (default true) emphasizes fine
    texture over overall density. Returns ranked matches with image ids, source
    slice indices, and pixel bounding boxes.
    """
    payload = {
        "collection": collection, "bbox": [row, col, h, w], "px": px,
        "topk": topk, "whiten": whiten,
        "image_id": image_id, "dataset_index": dataset_index,
    }
    with _client() as c:
        r = c.post("/query", json=payload)
        if r.status_code != 200:
            return {"error": r.text, "status": r.status_code}
        return r.json()


@mcp.tool()
def get_image_url(collection: str, image_id: int) -> dict:
    """Return the service URL of a stored image PNG (for display / cropping)."""
    return {"url": f"{SERVICE_URL}/image/{collection}/{image_id}.png"}


def run():
    mcp.run()


if __name__ == "__main__":
    run()
