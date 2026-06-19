"""tomojepa: self-supervised tomography representation learning toolkit.

Bundles the self-supervised pre-training and downstream subsystems that share a
common backbone and data loader:

- ``tomojepa.ssl``      LeJEPA / DINOv3 self-supervised pre-training + validation.
- ``tomojepa.swinjepa`` Swin multi-scale latent-JEPA pre-training (per-stage SIGReg,
                        no EMA teacher, no pixel decoder).
- ``tomojepa.vitup``    ViT-Up faithful feature upsampling (distillation + inference).
- ``tomojepa.patchdb``  FAISS + DuckDB patch-retrieval engine (CLI / service / MCP).

The shared library lives in ``tomojepa.core`` (model, dataset, augmentations).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
