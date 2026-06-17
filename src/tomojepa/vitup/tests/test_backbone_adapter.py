"""Backbone-interface test (acceptance criterion 1)."""
import torch


def test_structure(adapter):
    assert adapter.p == 16
    assert adapter.C == 384
    assert adapter.L == 12
    assert hasattr(adapter.patch_embed, "proj")


def test_hidden_states_and_centers(adapter):
    img = torch.randn(2, 1, 64, 64)
    hid = adapter.hidden_states(img, [0, 2, 4])
    for l in (0, 2, 4):
        assert hid[l].shape == (2, 384, 4, 4)        # 64/16 = 4 token grid
    centers = adapter.token_centers(4, 4)
    assert centers.shape == (4, 4, 2)
    assert torch.allclose(centers[0, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(centers[3, 3], torch.tensor([3.5, 3.5]))


def test_lora_freezes_base():
    from tomojepa.vitup.backbone_adapter import build_backbone, BackboneAdapter
    bb = build_backbone("vit_small_patch16_dinov3", in_chans=1, img_size=64)
    a = BackboneAdapter(bb)
    a.apply_lora(("patch_embed", "attn.qkv", "attn.proj"), r=16, alpha=32, dropout=0.05)
    has_lora = any(("lora_A" in n or "lora_B" in n) for n, _ in a.named_parameters())
    assert has_lora
    for n, p in a.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            assert p.requires_grad
        else:
            assert not p.requires_grad
