"""tomojepa: self-supervised tomography representation learning toolkit.

Bundles three subsystems that share a common ViT backbone and data loader:

- ``tomojepa.ssl``      LeJEPA / DINOv3 self-supervised pre-training + validation.
- ``tomojepa.vitup``    ViT-Up faithful feature upsampling (distillation + inference).
- ``tomojepa.patchdb``  FAISS + DuckDB patch-retrieval engine (CLI / service / MCP).

The shared library lives in ``tomojepa.core`` (model, dataset, augmentations).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
