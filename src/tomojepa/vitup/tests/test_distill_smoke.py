"""Smoke training test (acceptance criterion 7).

A few optimizer steps of the full distillation objective on random data should
reduce the loss. Uses two small random backbones (teacher + LoRA student) on CPU.
"""
import torch

from tomojepa.vitup.backbone_adapter import build_backbone, BackboneAdapter
from tomojepa.vitup.model import ViTUp
from tomojepa.vitup.distill import MultiScaleTeacher, DistillEngine
from .conftest import small_cfg


def test_smoke_train_decreases_loss():
    torch.manual_seed(0)
    cfg = small_cfg(query_chunk_size=8)

    teacher_bb = build_backbone("vit_small_patch16_dinov3", in_chans=1, img_size=64)
    teacher = MultiScaleTeacher(BackboneAdapter(teacher_bb),
                                list(cfg.teacher_resolutions),
                                layers=[0] + list(cfg.layer_indices))

    student_bb = build_backbone("vit_small_patch16_dinov3", in_chans=1, img_size=64)
    student = BackboneAdapter(student_bb)
    student.apply_lora(cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    vitup = ViTUp(student, cfg)
    engine = DistillEngine(teacher, vitup, cfg)

    trainable = [p for p in engine.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    torch.manual_seed(1)
    img = torch.randn(2, 1, 64, 64)
    vitup.train()
    losses = []
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        loss, logs = engine.compute_loss(img)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert all(l == l for l in losses)  # no NaNs
