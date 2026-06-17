"""ViT-Up training entrypoint -- multi-scale feature distillation.

Mirrors this repo's conventions ([main.py](main.py)): single-process / single-GPU,
``argparse`` config, AMP (bf16), AdamW + warmup/cosine, manual checkpoint
save/resume, and the existing ``TomographyDataset`` loader. A frozen teacher
backbone (``--teacher_ckpt``) supervises a LoRA-adapted student backbone +
ViT-Up.

Examples:
    # full run (paper: ImageNet 1 epoch, batch 24, lr 2e-4 cosine)
    python train_vitup.py --teacher_ckpt runs/soil_residual_fg/ckpt/ckpt_last.pth \
        --data_dir . --pattern 'soild_stack.zarr' --backend zarr \
        --dataset_key reconstruction --epochs 1 --batch_size 24 --lr 2e-4

    # fast ablation/smoke config (20k iters, batch 16, output grid 32)
    python train_vitup.py --teacher_ckpt <ckpt> --data_dir . --pattern '*.zarr' \
        --batch_size 16 --max_iters 20000 --query_grid 32
"""
import os
import re
import glob
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.dataset import TomographyDataset
from ..core import dist as D
from .config import add_argparse_args, from_args
from .backbone_adapter import build_backbone, load_backbone_state, BackboneAdapter
from .model import ViTUp
from .distill import MultiScaleTeacher, DistillEngine
from .infer import pca_maps


def parse_args():
    p = argparse.ArgumentParser(description="ViT-Up multi-scale distillation")
    # data
    p.add_argument("--data_dir", required=True, help="Directory of .h5/.zarr volumes")
    p.add_argument("--pattern", default="*.zarr", help="Glob for volume files")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="auto")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--augment", choices=["tomo", "tomo2"], default="tomo2")
    p.add_argument("--img_size", type=int, default=512, help="source training-image size")
    p.add_argument("--num_workers", type=int, default=8)
    # teacher / student
    p.add_argument("--teacher_ckpt", required=True,
                   help="DINOv3ViTEncoder checkpoint for the frozen teacher + student init")
    # io / logging
    p.add_argument("--ckpt_dir", default="checkpoints_vitup")
    p.add_argument("--out_dir", default="outputs_vitup")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    # periodic PCA progress probe (fixed slices, same basis layout as vitup_pca.py)
    p.add_argument("--pca_every", type=int, default=0,
                   help="run a PCA progress probe every N steps (0 = off)")
    p.add_argument("--pca_samples", type=int, default=2,
                   help="number of fixed slices in the PCA probe")
    p.add_argument("--pca_res", type=int, default=512,
                   help="backbone + ViT-Up output resolution for the probe")
    p.add_argument("--pca_chunk", type=int, default=32768,
                   help="query chunk size for the probe upsample")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="ViTUp_Tomo")
    # all ViTUpConfig fields (architecture, lora, train losses, optimization)
    add_argparse_args(p)
    return p.parse_args()


def save_ckpt(ckpt_dir, tag, epoch, step, engine, opt, sched):
    torch.save({"epoch": epoch, "step": step,
                "engine": engine.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict()},
               os.path.join(ckpt_dir, f"ckpt_{tag}.pth"))


def maybe_resume(ckpt_dir, engine, opt, sched, device):
    last = os.path.join(ckpt_dir, "ckpt_last.pth")
    cands = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pth"))
    if os.path.exists(last):
        path = last
    elif cands:
        path = max(cands, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    else:
        return 0, 0
    print(f"Resuming from {path}", flush=True)
    ckpt = torch.load(path, map_location=device)
    engine.load_state_dict(ckpt["engine"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    return ckpt["epoch"] + 1, ckpt["step"]


def build_pca_probe(args, device):
    """Load a fixed set of slices (deterministic) for the recurring PCA probe."""
    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant="tomo2", img_size=args.pca_res,
        is_train=False, backend=args.backend,
    )
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(ds), size=min(args.pca_samples, len(ds)),
                             replace=False).tolist())
    imgs = []
    for idx in idxs:
        img = ds[int(idx)][0].unsqueeze(0)                      # [1,1,S,S]
        img = F.interpolate(img, size=(args.pca_res, args.pca_res),
                            mode="bilinear", align_corners=False)
        imgs.append(img.to(device))
    return idxs, imgs


@torch.no_grad()
def run_pca_probe(vitup, cfg, probe, step, out_dir, res, chunk):
    """Render a compact [orig | backbone-bilinear | ViT-Up] PCA strip per slice."""
    idxs, imgs = probe
    was_training = vitup.training
    vitup.eval()
    n = len(imgs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n + 0.5), squeeze=False)
    for r, (idx, img) in enumerate(zip(idxs, imgs)):
        with autocast(img.device.type, dtype=torch.bfloat16, enabled=img.is_cuda):
            ctx = vitup.encode_image(img)
            low = ctx.hidden[cfg.layer_indices[-1]][0].permute(1, 2, 0)
            dense = vitup.upsample(img, res, res, chunk_size=chunk)[0]
        dense_rgb, low_rgb, _ = pca_maps(dense.float(), low.float(), n_comp=3)
        h = low.shape[0]
        panels = [(img[0, 0].cpu().numpy(), f"slice {idx}", "gray"),
                  (low_rgb, f"backbone L{cfg.layer_indices[-1]} ({h}x{h}->bilinear)", None),
                  (dense_rgb, f"ViT-Up ({res}x{res})", None)]
        for c, (im, title, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cmap)
            if r == 0:
                ax.set_title(title, fontsize=11)
            ax.axis("off")
    fig.suptitle(f"PCA progress @ step {step}", fontsize=13)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"pca_step{step:06d}.png")
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    if was_training:
        vitup.train()
    return out


def main():
    args = parse_args()
    cfg = from_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Multi-GPU is opt-in via torchrun (sets WORLD_SIZE>1); otherwise single-proc.
    device, _local_rank = D.init_distributed()
    ws = D.world_size()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if D.is_main():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
    log_wandb = args.wandb and D.is_main()
    if log_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config={**vars(args)})

    # --- data: one source image per sample ---------------------------------
    train_ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant=args.augment, img_size=args.img_size,
        is_train=True, backend=args.backend,
    )
    sampler = (DistributedSampler(train_ds, shuffle=True, drop_last=True)
               if D.is_distributed() else None)
    loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=(sampler is None),
                         drop_last=True, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"), sampler=sampler)
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(train_ds, **loader_kwargs)
    if D.is_main():
        ddp = f" x {ws} GPUs (global batch {cfg.batch_size * ws})" if D.is_distributed() else ""
        print(f"Dataset: {len(train_ds)} slices over {len(train_ds.files)} volume(s) "
              f"-> {len(loader)} steps/epoch on {device}{ddp}", flush=True)

    # --- teacher (frozen) + student (LoRA) ---------------------------------
    teacher_bb = build_backbone(cfg.backbone_name, cfg.input_channels,
                                cfg.backbone_img_size)
    load_backbone_state(teacher_bb, args.teacher_ckpt, device)
    teacher_adapter = BackboneAdapter(teacher_bb).to(device)

    student_bb = build_backbone(cfg.backbone_name, cfg.input_channels,
                                cfg.backbone_img_size)
    load_backbone_state(student_bb, args.teacher_ckpt, device)
    student_adapter = BackboneAdapter(student_bb)
    student_adapter.apply_lora(cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha,
                               cfg.lora_dropout)

    vitup = ViTUp(student_adapter, cfg).to(device)
    teacher = MultiScaleTeacher(teacher_adapter, list(cfg.teacher_resolutions),
                                layers=[0] + list(cfg.layer_indices))
    engine = DistillEngine(teacher, vitup, cfg).to(device)

    trainable = [p for p in engine.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    if D.is_main():
        print(f"Trainable params: {n_train/1e6:.2f}M (LoRA r={cfg.lora_rank} + ViT-Up "
              f"T={cfg.num_blocks}, D={vitup.internal_dim}); teacher frozen", flush=True)
        print(f"Teacher resolutions {list(cfg.teacher_resolutions)} -> grids "
              f"{cfg.teacher_token_grids(vitup.adapter.p)}; query_grid={cfg.query_grid}; "
              f"canvas={cfg.student_canvas}; chunk={cfg.query_chunk_size}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_iters > 0:
        total_steps = min(total_steps, cfg.max_iters)
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.amp_dtype]
    use_amp = (amp_dtype != torch.float32) and device.type == "cuda"

    start_epoch, global_step = maybe_resume(args.ckpt_dir, engine, opt, sched, device)

    pca_probe = None
    pca_dir = os.path.join(args.out_dir, "pca")
    if args.pca_every > 0 and D.is_main():          # probe/viz on rank 0 only
        pca_probe = build_pca_probe(args, device)
        print(f"PCA probe: slices {pca_probe[0]} -> {pca_dir} every "
              f"{args.pca_every} steps", flush=True)

    engine.teacher.adapter.eval()  # teacher always frozen/eval
    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        vitup.train()
        if sampler is not None:                     # reshuffle shards per epoch
            sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(total=steps_per_epoch, desc=f"epoch {epoch}") if D.is_main() else None
        for vs, _ in loader:
            img = vs[:, 0].to(device, non_blocking=True)       # [B,C,H,W]
            opt.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = engine.compute_loss(img, chunk_size=cfg.query_chunk_size)
            loss.backward()
            # Per-sample mean loss -> standard data-parallel gradient averaging.
            D.average_grads_(trainable)
            opt.step()
            sched.step()

            if D.is_main() and global_step % args.log_every == 0:
                post = {k: f"{v:.3f}" for k, v in logs.items() if k != "total"}
                post["loss"] = f"{logs['total']:.3f}"
                pbar.set_postfix(**post)
                if log_wandb:
                    wandb.log({f"train/{k}": v for k, v in logs.items()}
                              | {"train/lr": sched.get_last_lr()[0]}, step=global_step)
            if (D.is_main() and args.save_every and global_step > 0
                    and global_step % args.save_every == 0):
                save_ckpt(args.ckpt_dir, "last", epoch, global_step, engine, opt, sched)
            if pca_probe is not None and global_step % args.pca_every == 0:
                out = run_pca_probe(vitup, cfg, pca_probe, global_step, pca_dir,
                                    args.pca_res, args.pca_chunk)
                pbar.write(f"[pca] saved {out}")

            global_step += 1
            if pbar is not None:
                pbar.update(1)
            if cfg.max_iters > 0 and global_step >= cfg.max_iters:
                stop = True
                break
        if pbar is not None:
            pbar.close()
        if D.is_main():
            save_ckpt(args.ckpt_dir, f"epoch_{epoch}", epoch, global_step, engine, opt, sched)
            save_ckpt(args.ckpt_dir, "last", epoch, global_step, engine, opt, sched)
        if stop:
            break

    if log_wandb:
        wandb.finish()
    if D.is_main():
        print("done.", flush=True)
    D.cleanup()


if __name__ == "__main__":
    main()
