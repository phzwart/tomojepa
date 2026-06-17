"""patchdb: a FAISS-backed, DuckDB-persisted patch-retrieval engine.

Encodes images into shared-basis per-token codes, indexes the foreground tokens
in FAISS for fast candidate generation, and re-ranks candidate windows at the
query's exact size with an integral-image pooling pass. Exposed through a CLI
(`patchdb`), a FastAPI service, and an MCP tool wrapper.
"""
from .. import __version__

__all__ = ["__version__"]
