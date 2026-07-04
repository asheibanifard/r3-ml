"""siren.py — SIREN implicit field fitting for FAFB EM blocks.

Pure-CUDA implementation of Sitzmann et al., "Implicit Neural Representations
with Periodic Activation Functions" (NeurIPS 2020): a multi-layer perceptron
with sine activations, f(x) = sigmoid(W_L sin(w0 (... sin(w0 (W_0 x + b_0)) ...)) + b_L),
fit to the same 64×64×64 EM blocks as the 3DGS model for direct comparison.

Forward and backward (matmuls, bias-add, sine/sigmoid + their analytic
derivatives) are hand-written CUDA kernels (scripts/siren/siren_cuda.cu),
JIT-compiled via torch.utils.cpp_extension — no cuBLAS, no autograd-traced
PyTorch ops inside the network itself, matching the project's existing
3dgs_cuda.cu philosophy of a hand-written training kernel.

Data loading, sampling, and PSNR evaluation reuse scripts/_3dgs/_3dgs.py's
VolumeDataset / AABB / psnr_on_samples / vol_psnr so results are directly
comparable to the Gaussian-cloud model on the same blocks.

Usage
-----
  /venv/r3-ml/bin/python3 scripts/siren/siren.py \
    --volume data/fafb/blocks/image_z0_y0_x0.tif \
    --flat_out \
    --out models_siren/z000_y000_x000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _3dgs._3dgs import (
    AABB, VolumeDataset, evaluate_fields, _find_cuda_include, _load_yaml_config,
)
from _3dgs._3dgs_training import _LogFile, _load_volume, _visualize_middle_slices


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernel — lazy JIT compile, cached for the process lifetime
# ─────────────────────────────────────────────────────────────────────────────
_siren_cuda = None


def _load_siren_kernel():
    """Lazily compile and cache the hand-written SIREN forward/backward extension."""
    global _siren_cuda
    if _siren_cuda is None:
        src        = Path(__file__).parent / "siren_cuda.cu"
        extra_inc  = _find_cuda_include()
        extra_flags = ["-O3", "--use_fast_math"] + [f"-I{p}" for p in extra_inc]
        _siren_cuda = load(
            name="siren_cuda",
            sources=[str(src)],
            extra_cuda_cflags=extra_flags,
            extra_include_paths=extra_inc,
            verbose=False,
        )
    return _siren_cuda


# ─────────────────────────────────────────────────────────────────────────────
# Autograd wrapper around the fused CUDA kernel
# ─────────────────────────────────────────────────────────────────────────────
# Why *params instead of a list?  torch.autograd.Function only tracks Tensor
# arguments passed as direct positional args to .apply() — a Python list of
# tensors passed as a single arg is opaque to the autograd graph and would
# silently break gradient flow to individual weight/bias tensors. Flattening
# weights+biases into *params keeps every parameter a genuine tracked input.
#%%
class _SirenFn(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA SIREN forward/backward kernel."""

    @staticmethod
    def forward(ctx, pts, w0_first, w0_hidden, *params):
        L = len(params) // 2
        weights = list(params[0::2])
        biases  = list(params[1::2])

        kernel = _load_siren_kernel()
        result = kernel.forward(pts.contiguous(), weights, biases, w0_first, w0_hidden)

        pred = result[0]
        Zs   = result[1:L]
        acts = result[L:2 * L]

        ctx.save_for_backward(pred, *Zs, *acts, *weights)
        ctx.L = L
        ctx.w0_first  = w0_first
        ctx.w0_hidden = w0_hidden
        return pred

    @staticmethod
    def backward(ctx, grad_pred):
        L = ctx.L
        saved   = ctx.saved_tensors
        pred    = saved[0]
        Zs      = list(saved[1:L])
        acts    = list(saved[L:2 * L])
        weights = list(saved[2 * L:3 * L])

        kernel = _load_siren_kernel()
        grads = kernel.backward(
            grad_pred.contiguous(), pred, Zs, acts, weights,
            ctx.w0_first, ctx.w0_hidden,
        )

        dparams = []
        for i in range(L):
            dparams.append(grads[2 * i])       # dW_i
            dparams.append(grads[2 * i + 1])   # db_i
        return (None, None, None, *dparams)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def _siren_init_(weight: torch.Tensor, bias: torch.Tensor, in_features: int,
                 is_first: bool, omega_0: float):
    """Paper-faithful init (Sitzmann et al., Sec. 3.2 / supplement).

    First layer: U(-1/n, 1/n) so the pre-activation spans several periods of
    the sine once scaled by omega_0. Every later layer (including the final
    linear output, per the reference implementation): U(-sqrt(6/n)/omega_0,
    sqrt(6/n)/omega_0), which keeps the pre-activation distribution stable
    across depth. Biases keep PyTorch's default nn.Linear init (U(-1/sqrt(n),
    1/sqrt(n))) — the paper does not special-case them.
    """
    with torch.no_grad():
        bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / omega_0)
        weight.uniform_(-bound, bound)
        bias_bound = 1.0 / math.sqrt(in_features)
        bias.uniform_(-bias_bound, bias_bound)


class SIRENField(nn.Module):
    """Sinusoidal implicit field fitting a single EM block: R^3 -> [0,1].

    Architecture: 3 -> [hidden_dim -(sine)->]*hidden_layers -> 1 -(sigmoid)->.
    All linear algebra runs through the hand-written CUDA kernel in
    siren_cuda.cu via _SirenFn; this class only owns the parameters and the
    paper-faithful initialisation scheme.
    """

    def __init__(self, hidden_layers: int = 4, hidden_dim: int = 256,
                 w0_first: float = 30.0, w0_hidden: float = 30.0):
        super().__init__()
        assert hidden_layers >= 1, "need at least one sine layer before the output layer"
        self.hidden_layers = hidden_layers
        self.hidden_dim    = hidden_dim
        self.w0_first      = w0_first
        self.w0_hidden     = w0_hidden

        dims = [3] + [hidden_dim] * hidden_layers + [1]
        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()
        for l in range(len(dims) - 1):
            in_f, out_f = dims[l], dims[l + 1]
            is_first = (l == 0)
            omega    = w0_first if is_first else w0_hidden
            W = torch.empty(out_f, in_f)
            b = torch.empty(out_f)
            _siren_init_(W, b, in_f, is_first, omega)
            self.weights.append(nn.Parameter(W))
            self.biases.append(nn.Parameter(b))

    @property
    def device(self) -> torch.device:
        return self.weights[0].device

    def forward(self, pts: torch.Tensor, chunk_n: int | None = None) -> torch.Tensor:
        """Evaluate the field at query points, chunking over M to cap peak VRAM.

        chunk_n mirrors GaussianCloud.forward's signature (used identically by
        the reused psnr_on_samples / vol_psnr / _visualize_middle_slices
        helpers) but here it chunks the query-point batch rather than a
        Gaussian count — there is no analogous "primitive count" dimension
        in an MLP.
        """
        params = []
        for w, b in zip(self.weights, self.biases):
            params += [w, b]

        if chunk_n is None or pts.shape[0] <= chunk_n:
            return _SirenFn.apply(pts, self.w0_first, self.w0_hidden, *params)

        outs = []
        for s in range(0, pts.shape[0], chunk_n):
            e = min(s + chunk_n, pts.shape[0])
            outs.append(_SirenFn.apply(pts[s:e], self.w0_first, self.w0_hidden, *params))
        return torch.cat(outs, dim=0)

    def save(self, path):
        torch.save({
            "weights":       [w.detach().cpu() for w in self.weights],
            "biases":        [b.detach().cpu() for b in self.biases],
            "hidden_layers": self.hidden_layers,
            "hidden_dim":    self.hidden_dim,
            "w0_first":      self.w0_first,
            "w0_hidden":     self.w0_hidden,
        }, str(path))

    @classmethod
    def load(cls, path, device) -> "SIRENField":
        ckpt  = torch.load(str(path), map_location=device, weights_only=True)
        model = cls(ckpt["hidden_layers"], ckpt["hidden_dim"], ckpt["w0_first"], ckpt["w0_hidden"])
        with torch.no_grad():
            for w, w_ckpt in zip(model.weights, ckpt["weights"]):
                w.copy_(w_ckpt)
            for b, b_ckpt in zip(model.biases, ckpt["biases"]):
                b.copy_(b_ckpt)
        return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / LR schedule
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(pred: torch.Tensor, gt: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Per-sample-weighted MSE — the paper's loss for implicit field fitting."""
    err = (pred - gt) ** 2
    if sample_weights is not None:
        return (err * sample_weights).mean()
    return err.mean()


def update_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, cfg: argparse.Namespace):
    """Linear warmup over lr_warmup_steps, then cosine decay to lr_final_fraction·lr."""
    warmup = max(int(getattr(cfg, "lr_warmup_steps", 0)), 0)
    final_frac = float(getattr(cfg, "lr_final_fraction", 0.05))

    if warmup > 0 and step < warmup:
        scale = (step + 1) / warmup
    else:
        prog  = (step - warmup) / max(total_steps - warmup, 1)
        prog  = min(max(prog, 0.0), 1.0)
        scale = final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))

    for g in optimizer.param_groups:
        g["lr"] = cfg.lr * scale


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: argparse.Namespace):
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.flat_out:
        out_dir  = Path(cfg.out)
        ckpt_dir = out_dir
    else:
        out_dir  = Path(cfg.out) / f"{run_stamp}_siren_vol"
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train.log / log.json optionally routed to a separate logs directory
    # (mirroring train_all_blocks.py's logs_dir/models_dir split), while
    # checkpoints + config.json always stay together under out_dir.
    logs_dir = Path(cfg.logs_dir) if getattr(cfg, "logs_dir", None) else out_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    log = _LogFile(logs_dir / "train.log")
    log.write(f"# run  : {run_stamp}")
    log.write(f"# vol  : {cfg.volume}")
    log.write(f"# date : {datetime.now().isoformat(timespec='seconds')}")
    log.write(f"# arch : {cfg.hidden_layers} sine layers x {cfg.hidden_dim}, "
              f"w0_first={cfg.w0_first} w0_hidden={cfg.w0_hidden}")
    log.write(f"# {'epoch':>6}  {'step':>7}  {'loss':>9}  {'psnr':>7}  {'vol_psnr':>8}  {'elapsed':>8}")

    print(f"Loading volume: {cfg.volume}")
    volume, vmin, vmax = _load_volume(cfg.volume)
    d, h, w = volume.shape
    print(f"  Shape     : {d} x {h} x {w}  ({d*h*w/1e6:.1f} M voxels)")
    print(f"  Intensity : [{vmin:.4f}, {vmax:.4f}] -> [0, 1]")

    device  = torch.device(cfg.device)
    aabb    = AABB.unit()
    dataset = VolumeDataset(volume, aabb, cfg, swc_path=None)

    model = SIRENField(cfg.hidden_layers, cfg.hidden_dim, cfg.w0_first, cfg.w0_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    update_lr(optimizer, 0, total_steps, cfg)

    if cfg.flat_out:
        init_path  = out_dir / "init.pth"
        best_path  = out_dir / "best.pth"
        final_path = out_dir / "last.pth"
    else:
        init_path  = out_dir / f"init_{run_stamp}.pth"
        best_path  = out_dir / f"best_{run_stamp}.pth"
        final_path = out_dir / f"final_{run_stamp}.pth"
    model.save(init_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model     : {cfg.hidden_layers} sine layers x {cfg.hidden_dim}  ({n_params:,} params)")
    print(f"  Steps     : {total_steps}  ({cfg.epochs} ep x {cfg.steps_per_epoch})")

    log_entries = []
    best_psnr   = -float("inf")
    bad_epochs  = 0
    t0 = time()
    detail_interval = max(int(cfg.eval_detail_interval), 1)

    step = 0
    last_psnr = float("nan")
    epoch_bar = tqdm(range(cfg.epochs), desc="epoch", unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        epoch_loss = torch.zeros((), device=device)

        for _ in range(cfg.steps_per_epoch):
            step += 1
            pts, gt, sample_weights = dataset.sample(cfg.batch, device, cfg=cfg)

            optimizer.zero_grad()
            pred = model(pts, chunk_n=cfg.chunk_n)
            loss = compute_loss(pred, gt, sample_weights)
            loss.backward()
            optimizer.step()
            update_lr(optimizer, step, total_steps, cfg)

            epoch_loss = epoch_loss + loss.detach()

        avg_loss = (epoch_loss / cfg.steps_per_epoch).item()
        detail_eval = ((epoch + 1) % detail_interval == 0) or (epoch == cfg.epochs - 1)
        eval_metrics = evaluate_fields(model, dataset, cfg, detail=detail_eval)
        last_psnr = eval_metrics["psnr"]
        elapsed = time() - t0

        log_entries.append({
            "epoch": epoch + 1,
            "step": step,
            "loss": round(avg_loss, 6),
            "psnr": round(eval_metrics["psnr"], 4),
            "vol_psnr": None if math.isnan(eval_metrics["vol_psnr"]) else round(eval_metrics["vol_psnr"], 4),
            "elapsed_s": round(elapsed, 1),
        })

        ckpt_epoch_interval = max(int(getattr(cfg, "ckpt_epoch_interval", 10)), 1)
        if (epoch + 1) % ckpt_epoch_interval == 0 or epoch == cfg.epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pth"
            model.save(ckpt_path)
            tqdm.write(f"  [epoch {epoch+1:4d}] saved checkpoint -> {ckpt_path.name}")

            vis_path = _visualize_middle_slices(dataset.vol, model, out_dir, epoch + 1, device)
            if vis_path:
                tqdm.write(f"  [epoch {epoch+1:4d}] saved visualization -> {Path(vis_path).name}")

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", psnr=f"{last_psnr:.2f}", t=f"{elapsed/60:.1f}m")

        vol_psnr_str = ("     nan" if math.isnan(eval_metrics["vol_psnr"])
                        else f"{eval_metrics['vol_psnr']:8.3f}")
        log.write(
            f"  {epoch+1:6d}  {step:7d}  {avg_loss:9.6f}  {last_psnr:7.3f}  "
            f"{vol_psnr_str}  {elapsed/60:7.1f}m"
        )

        # Best-checkpoint selection uses vol_psnr (exact full-grid metric) to
        # match the 3DGS pipeline's convention — see CLAUDE.md.
        if detail_eval:
            this_vol_psnr = eval_metrics["vol_psnr"]
            if this_vol_psnr > best_psnr:
                best_psnr  = this_vol_psnr
                bad_epochs = 0
                model.save(best_path)
                best_msg = (f"  * new best  epoch {epoch+1:4d}  "
                            f"vol_PSNR={this_vol_psnr:.2f} dB  t={elapsed/60:.1f} min")
                tqdm.write(best_msg)
                log.write(f"# {best_msg.strip()}")
            else:
                bad_epochs += 1
                if cfg.early_stop_patience is not None and bad_epochs >= cfg.early_stop_patience:
                    tqdm.write(
                        f"  [epoch {epoch+1:4d}] early stopping after {bad_epochs} stagnant "
                        f"detail-eval epochs (best vol_PSNR={best_psnr:.2f} dB)"
                    )
                    break

    epoch_bar.close()
    model.save(final_path)

    with open(logs_dir / "log.json", "w") as f:
        json.dump(log_entries, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    total_min = (time() - t0) / 60
    log.write(f"# finished : {datetime.now().isoformat(timespec='seconds')}")
    log.write(f"# best vol_PSNR: {best_psnr:.3f} dB")
    log.write(f"# total    : {total_min:.1f} min")
    log.write(f"# models   : {out_dir}")
    log.write(f"# logs     : {logs_dir}")
    log.close()

    print(f"\nDone. Best vol_PSNR = {best_psnr:.2f} dB  ->  {best_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIREN implicit field fitting — volumetric regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="YAML config file; CLI flags override YAML values")

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--volume",   default=None)
    p.add_argument("--out",      default="models/siren/run",
                   help="checkpoints (init/best/last.pth) + config.json go here")
    p.add_argument("--logs_dir", default=None,
                   help="train.log + log.json go here instead of --out, if set")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Model (paper defaults: Sitzmann et al. 2020) ────────────────────────
    p.add_argument("--hidden_layers", type=int,   default=4, help="number of sine layers before the output layer")
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--w0_first",      type=float, default=30.0)
    p.add_argument("--w0_hidden",     type=float, default=30.0)

    # ── Training schedule ─────────────────────────────────────────────────────
    p.add_argument("--epochs",          type=int,   default=1000)
    p.add_argument("--steps_per_epoch", type=int,   default=50)
    p.add_argument("--batch",           type=int,   default=2048)
    p.add_argument("--chunk_n",         type=int,   default=200_000,
                   help="max query points per CUDA forward call (caps eval VRAM)")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="stop after this many detail-eval epochs without vol_PSNR improvement")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    p.add_argument("--lr",                 type=float, default=1e-4, help="paper default for Adam")
    p.add_argument("--lr_warmup_steps",    type=int,   default=100)
    p.add_argument("--lr_final_fraction",  type=float, default=0.05)

    # ── Eval ─────────────────────────────────────────────────────────────────
    p.add_argument("--eval_samples",         type=int, default=200_000)
    p.add_argument("--eval_detail_interval", type=int, default=5)
    p.add_argument("--eval_full_max_voxels", type=int, default=5_000_000)

    # ── Output layout ───────────────────────────────────────────────────────────
    p.add_argument("--flat_out",    dest="flat_out", action="store_true",
                   help="write checkpoints directly into --out (no timestamp subdir)")
    p.add_argument("--no_flat_out", dest="flat_out", action="store_false")
    p.set_defaults(flat_out=False)
    p.add_argument("--ckpt_epoch_interval", type=int, default=10)

    pre, _ = p.parse_known_args()
    if pre.config is not None:
        p.set_defaults(**_load_yaml_config(pre.config, p))

    args = p.parse_args()
    if args.volume is None:
        p.error("--volume is required (on CLI or in config file)")
    return args


if __name__ == "__main__":
    cfg = parse_args()
    print(f"Device : {cfg.device}")
    print(f"Output : {cfg.out}")
    train(cfg)
