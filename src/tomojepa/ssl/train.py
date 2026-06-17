"""LeJEPA / DINOv3 self-supervised pre-training on microCT tomography.

Single-process, single-GPU training intended for a local box (e.g. an NVIDIA
DGX Spark). No Slurm, no DDP, no Hydra -- just argparse.

Example:
    python main.py --data_dir /path/to/h5 --epochs 15 --batch_size 8

Resume is automatic: if ``checkpoints/ckpt_last.pth`` exists it is picked up.
"""
import os
import re
import glob
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.dataset import TomographyDataset
from ..core.model import (DINOv3ViTEncoder, SIGReg, MaskedLatentPredictor, encode_masked,
                          foreground_tokens, masked_mean)
from ..core import dist as D


def parse_args():
    p = argparse.ArgumentParser(description="LeJEPA DINOv3 tomography pre-training")
    # data
    p.add_argument("--data_dir", required=True, help="Directory of .h5/.zarr volumes")
    p.add_argument("--pattern", default="recon_*.h5", help="Glob for volume files")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="auto",
                   help="Storage backend; 'auto' infers from each file's extension")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--augment", choices=["tomo", "tomo2"], default="tomo2")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_chans", type=int, default=1,
                   help="1 for grayscale tomography, 3 for RGB baselines")
    # optimization
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=1,
                   help="microbatches accumulated per optimizer step via GradCache. "
                        "SIGReg is computed over the FULL effective batch "
                        "(batch_size * accum_steps), at single-microbatch memory.")
    p.add_argument("--global_views", type=int, default=2,
                   help="wide-area views per sample (scale band --global_scale)")
    p.add_argument("--local_views", type=int, default=2,
                   help="aggressively zoomed-in views per sample (scale band --local_scale)")
    p.add_argument("--global_scale", type=float, nargs=2, default=(0.4, 1.0),
                   metavar=("MIN", "MAX"), help="RandomResizedCrop area scale for global views")
    p.add_argument("--local_scale", type=float, nargs=2, default=(0.1, 0.4),
                   metavar=("MIN", "MAX"), help="RandomResizedCrop area scale for local views")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--lamb", type=float, default=0.02,
                   help="SIGReg vs. invariance weight (loss = lamb*sigreg + (1-lamb)*inv)")
    # masked latent prediction (MAE) + residual factorization
    p.add_argument("--mim_weight", type=float, default=0.0,
                   help="weight on the masked latent-prediction (MAE) loss. "
                        "0 disables MIM (pure LeJEPA on pooled emb, as before).")
    p.add_argument("--residual_local", action="store_true",
                   help="route LeJEPA invariance+SIGReg through the residual "
                        "z_local = proj(mean(T - sg(C))) instead of proj(emb), "
                        "making the invariant features complementary to the smooth "
                        "MAE context C. Requires --mim_weight > 0.")
    p.add_argument("--mask_ratio", type=float, default=0.5,
                   help="fraction of patch tokens masked for MIM")
    p.add_argument("--mask_blocks", type=int, default=4,
                   help="number of rectangular blocks for block masking")
    p.add_argument("--mim_target_norm", action="store_true", default=True,
                   help="per-token layer-norm the (stop-grad) MAE target")
    p.add_argument("--no_mim_target_norm", dest="mim_target_norm",
                   action="store_false")
    p.add_argument("--indep_weight", type=float, default=0.0,
                   help="optional cross-covariance (decorrelation) penalty between "
                        "z_local and the pooled smooth context, reinforcing the "
                        "structural conditional independence. 0 disables it.")
    p.add_argument("--foreground_mask", action="store_true",
                   help="restrict residual pooling and the MIM target to foreground "
                        "(sample-ROI) tokens, so capacity is not spent on the flat "
                        "background/frame. Detected per-view from patch intensity std.")
    p.add_argument("--fg_std_thresh", type=float, default=0.05,
                   help="per-patch intensity-std threshold for the foreground mask "
                        "(view units, ~[-1,1] after normalize)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    # logging / io
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--viz_every", type=int, default=1000, help="0 disables PCA viz")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--probe", action="store_true",
                   help="Train an online linear probe. Needs REAL labels in the "
                        "dataset; the current loader emits dummy 0 labels, so this "
                        "is off by default (it would just learn to predict class 0).")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb_project", default="LeJEPA_DINOv3_Tomo")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def generate_pca_viz(net, dataset, num_samples=3):
    """RGB-composite of the top-3 PCA components of patch tokens, vs. the slice."""
    device = next(net.parameters()).device
    figs = []
    step = max(1, len(dataset) // max(1, num_samples))
    for i in range(num_samples):
        try:
            views, _ = dataset[i * step]
            view = views[0].unsqueeze(0).to(device)                 # [1, C, H, W]
            feat = net.backbone.forward_features(view)              # [1, L, D]
            tokens = feat[:, net.backbone.num_prefix_tokens:].squeeze(0).float()
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            grid = int(np.sqrt(tokens.shape[0]))

            _, _, v = torch.pca_lowrank(tokens, q=3)
            pcs = (tokens @ v[:, :3]).reshape(grid, grid, 3).permute(2, 0, 1)[None]
            pca_img = F.interpolate(pcs, size=(512, 512), mode="bilinear",
                                    align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
            for c in range(3):
                lo, hi = np.percentile(pca_img[..., c], [1, 99])
                pca_img[..., c] = (np.clip((pca_img[..., c] - lo) / (hi - lo), 0, 1)
                                   if hi > lo else 0.0)

            orig = view.squeeze(0)[0].cpu().numpy()
            fig = plt.figure(figsize=(25, 5))
            plt.subplot(1, 5, 1); plt.imshow(orig, cmap="gray")
            plt.title(f"slice {i}"); plt.axis("off")
            plt.subplot(1, 5, 2); plt.imshow(pca_img)
            plt.title("PCA RGB"); plt.axis("off")
            for j in range(3):
                plt.subplot(1, 5, 3 + j); plt.imshow(pca_img[..., j], cmap="viridis")
                plt.title(f"PC {j + 1}"); plt.axis("off")
            plt.tight_layout()
            figs.append(fig)
        except Exception as e:
            print(f"[pca] sample {i} failed: {e}", flush=True)
    return figs


def lejepa_loss_terms(proj, sigreg, lamb):
    """LeJEPA loss on projected embeddings ``proj`` of shape [V, N, D].

    Returns ``(total, sigreg_value, invariance_value)`` where
    ``total = lamb*SIGReg + (1-lamb)*invariance``. SIGReg is a distribution-level
    statistic over the N samples, so it must be evaluated on the full batch
    (not summed across minibatches) -- which is exactly why accumulation uses
    GradCache rather than naive gradient accumulation.
    """
    inv = (proj.mean(0) - proj).square().mean()
    sg = sigreg(proj)
    total = sg * lamb + inv * (1.0 - lamb)
    return total, sg, inv


def make_block_mask(n, grid, ratio, blocks, device):
    """BEiT-style block mask: ``[n, grid*grid]`` bool, True where masked.

    Lays down up to ``blocks`` random rectangular blocks per sample until the
    target masked fraction is reached, then guarantees at least one masked and
    one visible token. Uses torch RNG so it is reproducible under RNG restore,
    but callers should cache the returned mask and reuse it across the two
    GradCache passes rather than regenerate.
    """
    p = grid * grid
    target = max(1, min(p - 1, int(round(ratio * p))))
    mask = torch.zeros(n, grid, grid, dtype=torch.bool, device=device)
    for b in range(n):
        filled = 0
        guard = 0
        while filled < target and guard < 10 * max(1, blocks):
            guard += 1
            max_area = target - filled
            area = max(1, max_area // max(1, blocks))
            log_ar = torch.empty(1, device=device).uniform_(math.log(0.3),
                                                            math.log(1.0 / 0.3))
            ar = float(log_ar.exp())
            h = min(max(int(round(math.sqrt(area * ar))), 1), grid)
            w = min(max(int(round(math.sqrt(area / ar))), 1), grid)
            top = int(torch.randint(0, grid - h + 1, (1,), device=device))
            left = int(torch.randint(0, grid - w + 1, (1,), device=device))
            mask[b, top:top + h, left:left + w] = True
            filled = int(mask[b].sum())
        flat = mask[b].view(-1)
        if filled >= p:                       # keep >=1 visible
            flat[0] = False
        elif filled == 0:                     # keep >=1 masked
            flat[0] = True
    return mask.view(n, p)


def residual_view(net, mim, view, mask, fg=None, target_norm=True, want_ctx_pool=False):
    """Compute the residual local embedding and the MAE loss for one view.

    Returns ``(z_local, mae, ctx_pool)`` where:
      - ``T`` = full (unmasked) patch-token latents of ``view`` (carries grad).
      - ``C`` = smooth context field from the masked encode + predictor head.
      - ``mae`` = smooth-L1 between ``C`` and the stop-grad (normed) ``T`` over
        masked positions.
      - ``z_local`` = ``net.proj(mean_p(T - sg(C)))`` -- the augmentation-invariant
        residual sitting on top of the smooth context.
      - ``ctx_pool`` = detached pooled ``C`` (only if ``want_ctx_pool``), for the
        optional decorrelation penalty.
    """
    prefix = net.backbone.num_prefix_tokens
    T = net.backbone.forward_features(view)[:, prefix:]            # [B, P, D], grad
    ctx = encode_masked(net.backbone, view, mask, mim.mask_token)  # [B, P, D]
    C = mim(ctx)                                                   # smooth field
    tgt = F.layer_norm(T, (T.size(-1),)) if target_norm else T
    tgt = tgt.detach()
    R = T - C.detach()                                            # residual
    if fg is not None:
        loss_pos = mask & fg                                      # predict masked foreground
        if not loss_pos.any():
            loss_pos = mask
        mae = F.smooth_l1_loss(C[loss_pos], tgt[loss_pos])
        z_local = net.proj(masked_mean(R, fg))                    # foreground pooling
        ctx_pool = masked_mean(C.detach(), fg) if want_ctx_pool else None
    else:
        mae = F.smooth_l1_loss(C[mask], tgt[mask])
        z_local = net.proj(R.mean(dim=1))                         # [B, proj_dim]
        ctx_pool = C.detach().mean(1) if want_ctx_pool else None
    return z_local, mae, ctx_pool


def decorr(z_local, z_ctx):
    """Squared Frobenius norm of the cross-covariance between two embeddings.

    Drives ``z_local`` and the pooled context toward (linear) independence,
    reinforcing the structural conditional independence of the residual.
    """
    zl = z_local - z_local.mean(0, keepdim=True)
    zc = z_ctx - z_ctx.mean(0, keepdim=True)
    n = max(1, zl.size(0) - 1)
    cov = (zl.transpose(0, 1) @ zc) / n                          # [Dl, Dc]
    return cov.square().sum()


def save_ckpt(ckpt_dir, tag, epoch, step, net, probe, opt, mim=None):
    ckpt = {"epoch": epoch, "step": step,
            "net": net.state_dict(), "opt": opt.state_dict()}
    if probe is not None:
        ckpt["probe"] = probe.state_dict()
    if mim is not None:
        ckpt["mim"] = mim.state_dict()
    torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_{tag}.pth"))


def maybe_resume(ckpt_dir, net, probe, opt, device, mim=None):
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
    net.load_state_dict(ckpt["net"])
    if probe is not None and "probe" in ckpt:
        probe.load_state_dict(ckpt["probe"])
    if mim is not None and "mim" in ckpt:
        mim.load_state_dict(ckpt["mim"])
    try:
        opt.load_state_dict(ckpt["opt"])
    except ValueError:
        # e.g. --probe was toggled since the checkpoint was written, so the
        # param-group layout differs. Keep the (loaded) weights, reset opt state.
        print("  [resume] optimizer layout changed; resuming weights only.", flush=True)
    return ckpt["epoch"] + 1, ckpt["step"]


def main():
    args = parse_args()
    args.V = args.global_views + args.local_views   # total views per sample
    if args.V < 1:
        raise ValueError("global_views + local_views must be >= 1")
    if args.residual_local and args.mim_weight <= 0:
        raise ValueError("--residual_local requires --mim_weight > 0.")
    if args.indep_weight > 0 and not args.residual_local:
        raise ValueError("--indep_weight requires --residual_local (the "
                         "decorrelation penalty is defined between z_local and "
                         "the pooled smooth context).")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Multi-GPU is opt-in via torchrun (sets WORLD_SIZE>1); otherwise single-proc.
    device, _local_rank = D.init_distributed()
    ws = D.world_size()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if D.is_distributed():
        if args.probe:
            raise ValueError("--probe is not supported with multi-GPU (WORLD_SIZE>1).")
        if args.amp_dtype == "fp16":
            raise ValueError("--amp_dtype fp16 is not supported with multi-GPU "
                             "(GradScaler state differs per rank); use bf16 or fp32.")

    if D.is_main():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)

    log_wandb = args.wandb and D.is_main()
    if log_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # --- data ---------------------------------------------------------------
    train_ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=args.global_views, local_views=args.local_views,
        global_scale=tuple(args.global_scale), local_scale=tuple(args.local_scale),
        variant=args.augment, img_size=args.img_size, is_train=True,
        backend=args.backend,
    )
    # Under torchrun, DistributedSampler shards the dataset across ranks; each
    # rank gets an equal, disjoint slice (drop_last keeps shard sizes constant,
    # which all_gather requires). Otherwise plain shuffling.
    sampler = (DistributedSampler(train_ds, shuffle=True, drop_last=True)
               if D.is_distributed() else None)
    loader_kwargs = dict(
        batch_size=args.batch_size, shuffle=(sampler is None), drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        sampler=sampler,
    )
    if args.num_workers > 0:
        # The model initializes CUDA in this process before the loader spins up
        # workers. The default 'fork' start method then copies a process with a
        # live CUDA context and the workers deadlock (hang in futex_wait). 'spawn'
        # gives each worker a clean interpreter and avoids it.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(train_ds, **loader_kwargs)
    eff_batch = args.batch_size * args.accum_steps
    if D.is_main():
        ddp = f" x {ws} GPUs (global batch {args.batch_size * ws})" if D.is_distributed() else ""
        print(f"Dataset: {len(train_ds)} slices over {len(train_ds.files)} volume(s) "
              f"-> {len(loader)} steps/epoch on {device}{ddp}", flush=True)
        print(f"Views: {args.global_views} global {tuple(args.global_scale)} + "
              f"{args.local_views} local {tuple(args.local_scale)} = {args.V}/sample", flush=True)
        if args.accum_steps > 1:
            print(f"GradCache: {args.accum_steps} microbatches x {args.batch_size} = "
                  f"effective batch {eff_batch} samples ({eff_batch * args.V} views) for SIGReg, "
                  f"at {args.batch_size}-sample memory", flush=True)

    # --- model / opt --------------------------------------------------------
    net = DINOv3ViTEncoder(proj_dim=args.proj_dim, img_size=args.img_size,
                           in_chans=args.in_chans, pretrained=False).to(device)
    sigreg = SIGReg().to(device)

    param_groups = [{"params": net.parameters(), "lr": args.lr,
                     "weight_decay": args.weight_decay}]
    mim = None
    grid = args.img_size // net.backbone.patch_embed.patch_size[0]
    if args.mim_weight > 0:
        mim = MaskedLatentPredictor(net.embed_dim).to(device)
        param_groups.append({"params": mim.parameters(), "lr": args.lr,
                             "weight_decay": args.weight_decay})
        if D.is_main():
            print(f"MIM: mask_ratio={args.mask_ratio} blocks={args.mask_blocks} on a "
                  f"{grid}x{grid} token grid; residual_local={args.residual_local}, "
                  f"indep_weight={args.indep_weight}", flush=True)
    probe = None
    if args.probe:
        probe = nn.Sequential(nn.LayerNorm(net.embed_dim),
                              nn.Linear(net.embed_dim, 10)).to(device)
        param_groups.append({"params": probe.parameters(),
                             "lr": 1e-3, "weight_decay": 1e-7})
    opt = torch.optim.AdamW(param_groups)
    # Params whose gradients are SUM-all-reduced across ranks each step (the
    # manual replacement for DDP's autograd-hook sync). Materialized as a list
    # because net.parameters() is a one-shot generator already consumed by opt.
    reduce_params = (list(net.parameters())
                     + (list(mim.parameters()) if mim is not None else []))

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.amp_dtype]
    use_amp = (amp_dtype != torch.float32) and device.type == "cuda"
    # GradScaler is only needed for fp16; bf16 has fp32's dynamic range.
    scaler = GradScaler(device.type, enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    accum = max(1, args.accum_steps)
    if accum > 1 and amp_dtype == torch.float16:
        raise ValueError("--accum_steps > 1 is not supported with fp16 (GradScaler + "
                         "two-pass GradCache); use bf16 or fp32.")
    if accum > 1 and probe is not None:
        raise ValueError("--probe is not supported with --accum_steps > 1.")

    # LR schedule is measured in optimizer steps (one per `accum` microbatches).
    steps_per_epoch = max(1, len(loader) // accum)
    warmup_steps = steps_per_epoch
    total_steps = max(2, steps_per_epoch * args.epochs)
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-3)
    scheduler = SequentialLR(opt, [s1, s2], milestones=[warmup_steps])

    start_epoch, global_step = maybe_resume(args.ckpt_dir, net, probe, opt, device, mim)
    for _ in range(global_step):                       # fast-forward LR schedule
        scheduler.step()

    def make_masks(n, n_views):
        return [make_block_mask(n, grid, args.mask_ratio, args.mask_blocks, device)
                for _ in range(n_views)]

    def view_fg(view):
        return (foreground_tokens(view, grid, args.fg_std_thresh)
                if args.foreground_mask else None)

    def residual_proj(vs, masks, want_indep):
        """Per-view residual embeddings -> ([V,N,proj], summed mae, summed indep)."""
        z_list, mae_acc, indep_acc = [], 0.0, 0.0
        for vi in range(args.V):
            z, m, ctx = residual_view(net, mim, vs[:, vi], masks[vi],
                                      fg=view_fg(vs[:, vi]),
                                      target_norm=args.mim_target_norm,
                                      want_ctx_pool=want_indep)
            z_list.append(z)
            mae_acc = mae_acc + m
            if want_indep:
                indep_acc = indep_acc + decorr(z, ctx)
        return torch.stack(z_list, 0), mae_acc / args.V, (
            indep_acc / args.V if want_indep else torch.zeros((), device=device))

    def mae_only(vs, masks):
        """MAE loss on the global views (additive MIM without the residual route)."""
        mae_acc = 0.0
        for vi in range(args.global_views):
            _, m, _ = residual_view(net, mim, vs[:, vi], masks[vi],
                                    fg=view_fg(vs[:, vi]),
                                    target_norm=args.mim_target_norm)
            mae_acc = mae_acc + m
        return mae_acc / max(1, args.global_views)

    def simple_step(vs, y):
        """One optimizer step on a single batch (no accumulation).

        Under multi-GPU, the local projections are gathered across ranks so the
        LeJEPA loss sees the global batch; per-sample MIM/decorr terms are scaled
        by ``1/world_size`` and the resulting parameter-grad partials are summed
        across ranks before the optimizer step (see ``tomojepa.ssl.distributed``).
        """
        vs = vs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        n = vs.size(0)
        opt.zero_grad(set_to_none=True)
        mae = torch.zeros((), device=device)
        indep = torch.zeros((), device=device)
        probe_loss = torch.zeros((), device=device)
        with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            emb = None
            if args.residual_local:
                masks = make_masks(n, args.V)
                proj, mae, indep = residual_proj(vs, masks, args.indep_weight > 0)
            else:
                emb, proj = net(vs)
                if mim is not None:
                    mae = mae_only(vs, make_masks(n, args.global_views))
            proj_global = D.all_gather_cat(proj, dim=1)        # [V, N*ws, D]
            D.sync_rng(args.seed + global_step)                # identical sketches
            total, sg, inv = lejepa_loss_terms(proj_global, sigreg, args.lamb)
            loss = total + D.loss_scale() * (args.mim_weight * mae
                                             + args.indep_weight * indep)
            if probe is not None:
                if emb is None:
                    emb = net(vs)[0]
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y.repeat_interleave(args.V))
                loss = loss + probe_loss
        scaler.scale(loss).backward()
        D.all_reduce_grads_(reduce_params)
        scaler.step(opt)
        scaler.update()
        return (total.item(), sg.item(), inv.item(), probe_loss.item(),
                float(mae.detach()), float(indep.detach()))

    def gradcache_step(micro):
        """One optimizer step over ``len(micro)`` microbatches via GradCache.

        Pass 1 (no grad): forward every microbatch, concatenate the projections,
        and evaluate the full-batch LeJEPA loss to get d(loss)/d(proj). SIGReg
        therefore sees ALL samples jointly (the whole point). Pass 2: re-forward
        each microbatch -- restoring the SAME RNG state so stochastic-depth masks
        match pass 1 -- and backprop the cached per-sample projection gradients.
        Net result: exact full-batch gradients at single-microbatch memory.

        With MIM the projection is the residual ``z_local`` (built from two
        backbone forwards per view: full ``T`` + masked context). The MAE loss is
        per-sample, so it is back-propagated per microbatch in pass 2 (scaled by
        1/accum) and accumulates into the same params alongside the cached LeJEPA
        residual gradient. Masks are generated once in pass 1 and reused in pass 2.
        """
        accum_n = len(micro)
        # Per-sample (MAE/decorr) terms are local-shard means; scale by
        # 1/(accum * world_size) so the SUM-all-reduce yields the global mean grad.
        mae_scale = D.loss_scale() / accum_n
        want_indep = args.indep_weight > 0
        projs, rng, mask_store = [], [], []
        with torch.no_grad(), autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            for vs, _ in micro:
                vs = vs.to(device, non_blocking=True)
                n = vs.size(0)
                if args.residual_local:
                    masks = make_masks(n, args.V)
                elif mim is not None:
                    masks = make_masks(n, args.global_views)
                else:
                    masks = None
                mask_store.append(masks)
                rng.append((torch.get_rng_state(),
                            torch.cuda.get_rng_state(device) if device.type == "cuda" else None))
                if args.residual_local:
                    proj, _, _ = residual_proj(vs, masks, False)
                else:
                    _, proj = net(vs)
                projs.append(proj.float())
        sizes = [p.size(1) for p in projs]
        proj_all = torch.cat(projs, dim=1).detach().requires_grad_(True)   # [V, N_total, D]
        # Multi-GPU: extend the effective batch across ranks too. The gather's
        # backward returns only this rank's slice, so grad_all is the local
        # gradient; per-rank param-grad partials are summed before opt.step().
        proj_global = D.all_gather_cat(proj_all, dim=1)
        D.sync_rng(args.seed + global_step)
        total, sg, inv = lejepa_loss_terms(proj_global, sigreg, args.lamb)
        total.backward()
        grad_all = proj_all.grad

        opt.zero_grad(set_to_none=True)
        offset = 0
        mae_acc, indep_acc = 0.0, 0.0
        for (vs, _), (cpu_state, cuda_state), masks, n in zip(micro, rng, mask_store, sizes):
            vs = vs.to(device, non_blocking=True)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            g = grad_all[:, offset:offset + n, :]
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                if args.residual_local:
                    proj, mae_mb, indep_mb = residual_proj(vs, masks, want_indep)
                    surrogate = (proj * g.to(proj.dtype)).sum()
                    surrogate = surrogate + (args.mim_weight * mae_scale) * mae_mb
                    if want_indep:
                        surrogate = surrogate + (args.indep_weight * mae_scale) * indep_mb
                    surrogate.backward()
                    mae_acc += float(mae_mb.detach())
                    indep_acc += float(indep_mb.detach()) if want_indep else 0.0
                elif mim is not None:
                    _, proj = net(vs)
                    mae_mb = mae_only(vs, masks)
                    surrogate = ((proj * g.to(proj.dtype)).sum()
                                 + (args.mim_weight * mae_scale) * mae_mb)
                    surrogate.backward()
                    mae_acc += float(mae_mb.detach())
                else:
                    _, proj = net(vs)
                    proj.backward(g.to(proj.dtype))
            offset += n
        D.all_reduce_grads_(reduce_params)
        opt.step()
        return (total.item(), sg.item(), inv.item(), 0.0,
                mae_acc / accum_n, indep_acc / accum_n)

    # --- train --------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        net.train()
        if probe is not None:
            probe.train()
        if mim is not None:
            mim.train()
        if sampler is not None:                     # reshuffle shards per epoch
            sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(total=steps_per_epoch, desc=f"epoch {epoch}") if D.is_main() else None
        buf = []
        for vs, y in loader:
            buf.append((vs, y))
            if len(buf) < accum:
                continue

            if accum == 1:
                lejepa_v, sigreg_v, inv_v, probe_v, mae_v, indep_v = simple_step(
                    buf[0][0], buf[0][1])
            else:
                lejepa_v, sigreg_v, inv_v, probe_v, mae_v, indep_v = gradcache_step(buf)
            buf = []
            scheduler.step()

            if D.is_main() and global_step % args.log_every == 0:
                logs = {"lejepa": lejepa_v, "sigreg": sigreg_v,
                        "inv": inv_v, "lr": scheduler.get_last_lr()[0]}
                if probe is not None:
                    logs["probe"] = probe_v
                post = dict(lejepa=f"{lejepa_v:.3f}", sigreg=f"{sigreg_v:.3f}")
                if mim is not None:
                    logs["mae"] = mae_v
                    post["mae"] = f"{mae_v:.3f}"
                    if args.indep_weight > 0:
                        logs["indep"] = indep_v
                pbar.set_postfix(**post)
                if log_wandb:
                    wandb.log({f"train/{k}": v for k, v in logs.items()}, step=global_step)

            if D.is_main() and args.viz_every and global_step % args.viz_every == 0:
                net.eval()
                for j, fig in enumerate(generate_pca_viz(net, train_ds)):
                    if log_wandb:
                        wandb.log({"train/pca": wandb.Image(fig)}, step=global_step)
                    fig.savefig(os.path.join(args.out_dir, f"pca_step{global_step}_{j}.png"),
                                dpi=80, bbox_inches="tight")
                    plt.close(fig)
                net.train()

            if (D.is_main() and args.save_every and global_step > 0
                    and global_step % args.save_every == 0):
                save_ckpt(args.ckpt_dir, "last", epoch, global_step, net, probe, opt, mim)

            global_step += 1
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        if D.is_main():
            save_ckpt(args.ckpt_dir, f"epoch_{epoch}", epoch, global_step, net, probe, opt, mim)
            save_ckpt(args.ckpt_dir, "last", epoch, global_step, net, probe, opt, mim)

    if log_wandb:
        wandb.finish()
    if D.is_main():
        print("done.", flush=True)
    D.cleanup()


if __name__ == "__main__":
    main()
