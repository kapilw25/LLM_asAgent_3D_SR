#!/usr/bin/env python3
"""m14 — VISUAL DEMO of the 4 headline eval metrics on real WalkIndia clips (iter19 task2.2).

Renders a humanly-understandable video (demo.mp4 + contact sheet for the visual-audit agent)
showing HOW each metric detects prediction quality — optionally as an A/B comparison
(e.g. FROZEN V-JEPA 2.1 vs OURS surgery-diheavy, both 1B) with models loaded SEQUENTIALLY so the
whole demo fits a 12 GB RTX 3060:

  Scene A  dense features          — per-slot PCA→RGB of encoder tokens beside the real frames
                                     (gold: facebookresearch/vjepa2 — "temporally consistent dense
                                     features, as highlighted by PCA visualizations").
  Scene B  future-frame MSE        — random spatio-temporal block masks (the SAME _MaskGenerator as
                                     m12d via utils.predictor_eval.build_mask_gen, same sample for
                                     every model); per-patch |pred − target| heatmaps side by side.
  Scene C  causal future-block L1  — first temporal half visible, second half predicted (the EXACT
                                     utils.pt_causal mask recipe); per-slot heatmaps + error curves.
  Scene D  mask-ratio robustness   — sweep r ∈ pipeline.yaml eval.predictor_temporal.mask_ratios
                                     with the SAME fixed-seed partition as utils.pt_maskratio;
                                     one L1-vs-r line per model, whose OLS slope IS the metric.
                                     The demo's sweep is asserted equal to utils.pt_maskratio.compute
                                     (FAIL LOUD parity guard — no re-implementation drift).
  Scene E  motion-cosine sep.      — pooled-feature cosine matrix per model across the demo clips;
                                     same-tour-type mean cos − cross-type mean cos (m12b margin
                                     analog at demo scale).

Metric cores are IMPORTED from utils.predictor_eval / utils.pt_causal / utils.pt_maskratio — the
demo visualizes the SAME computations the eval suite scores.

Gold-standard refs:
  https://github.com/facebookresearch/vjepa2                 (V-JEPA 2/2.1 + PCA dense-feature viz)
  https://arxiv.org/abs/2404.08471                           (V-JEPA — masked latent prediction)
  https://arxiv.org/abs/2203.12602                           (VideoMAE — high-mask-ratio robustness)

Decode substitution (documented): utils.video_io decodes via PyAV, which is NOT installed in the
demo venv (venv_walkindia); frames here are decoded with the system ffmpeg CLI (all frames →
uniform index sampling, identical semantics), then preprocessed with the EXACT eval recipe
utils.frozen_features.resize_and_normalize.

USAGE (RTX 3060 12 GB, bf16, B=1, models loaded one at a time):
    source venv_walkindia/bin/activate
    PYTHONPATH=src python -u src/m14_metric_demo.py \\
        --ckpt "FROZEN 2.1=checkpoints/vjepa2_1_vitg_384.pt" \\
        --ckpt "OURS diheavy=outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt" \\
        --model-config configs/model/vjepa2_1_vitg.yaml \\
        --demo-config configs/demo.yaml \\
        --clips-dir data/demo_clips \\
        --output-dir outputs/demo/metric_visual \\
        2>&1 | tee logs/m14_metric_demo_$(date +%Y%m%d_%H%M%S).log
    (single-model demo: pass --ckpt once)
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.config import get_pipeline_config
from utils.frozen_features import IMAGENET_MEAN, IMAGENET_STD, resize_and_normalize
from utils.predictor_eval import (
    PT_MASK_RATIOS, PT_SEED, build_mask_gen, expand_mask, load_encoder_predictor,
    masked_predict_l1, perclip_slope, temporal_token_idx, to_pixel, token_grid)
from utils import pt_causal, pt_maskratio

_HEAT_CMAP = "inferno"
_MODEL_COLORS = ("#90A4AE", "#2E7D32")     # model 1 = baseline grey · model 2 = OURS green


# ── Decode (ffmpeg CLI — PyAV substitution, see module docstring) ──────────────

def decode_uniform_frames(mp4: Path, num_frames: int) -> np.ndarray:
    """(num_frames, H, W, 3) uint8 — decode ALL frames via an ffmpeg rawvideo pipe, then take
    uniform indices (mirrors utils.video_io's uniform sampling)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    W, H = int(st["width"]), int(st["height"])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    if n == 0:
        sys.exit(f"FATAL: 0 frames decoded from {mp4}")
    arr = np.frombuffer(raw, np.uint8)[: n * W * H * 3].reshape(n, H, W, 3)
    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    return arr[idx].copy()


def denorm_to_uint8(clip_t: torch.Tensor) -> np.ndarray:
    """(T,3,crop,crop) ImageNet-normalized fp32 → (T,crop,crop,3) uint8 for display — the display
    pixels are EXACTLY the model-input pixels (same resize+crop), just de-normalized."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (clip_t * std + mean).clamp(0, 1)
    return (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


# ── Token-grid helpers ──────────────────────────────────────────────────────────

def tokens_to_grid(values: np.ndarray, idx: np.ndarray, num_frames: int) -> np.ndarray:
    """Scatter per-token values (aligned with sorted token indices `idx`) into (Tp,Hp,Wp);
    untouched cells = NaN."""
    Tp, Hp, Wp, S = token_grid(num_frames)
    flat = np.full(Tp * S, np.nan, dtype=np.float32)
    flat[idx] = values
    return flat.reshape(Tp, Hp, Wp)


def upsample_grid(g2d: np.ndarray, crop: int) -> np.ndarray:
    Hp, Wp = g2d.shape
    return np.repeat(np.repeat(g2d, crop // Hp, axis=0), crop // Wp, axis=1)


def pca_rgb(tokens: np.ndarray, num_frames: int, n_comp: int) -> np.ndarray:
    """tokens (N, D) fp32 → (Tp, Hp, Wp, 3) PCA→RGB in [0,1] (the vjepa2-repo dense-feature viz)."""
    Tp, Hp, Wp, _ = token_grid(num_frames)
    X = tokens - tokens.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:n_comp].T
    P = (P - P.min(0)) / (P.max(0) - P.min(0) + 1e-8)
    return P.reshape(Tp, Hp, Wp, n_comp)


# ── Per-model metric/visual payload (models loaded SEQUENTIALLY) ────────────────

def compute_payload(label, ckpt, model_config, batch, keys, heroes, num_frames, shared_masks):
    """Load ONE model, compute every visual + metric ingredient, free the GPU. Returns dict."""
    print(f"[m14] loading {label} ({ckpt}) …")
    encoder, predictor, _ = load_encoder_predictor(str(ckpt), num_frames, model_cfg=str(model_config))
    embed_dim = yaml.safe_load(Path(model_config).read_text())["model"]["embed_dim"]
    Tp, Hp, Wp, S = token_grid(num_frames)
    n_tok = Tp * S
    pl = {"label": label, "feats_final": {}, "pooled": {}, "hero": {}}

    for i, k in enumerate(tqdm(keys, desc=f"encode[{label}]")):
        pixel = to_pixel(batch[i:i + 1])
        with torch.no_grad():
            h = encoder(pixel)
            if isinstance(h, (list, tuple)):
                h = torch.cat(list(h), dim=-1)
        hf = h[0].float().cpu().numpy()
        pl["feats_final"][k] = hf[:, -embed_dim:]           # FINAL layer (Meta eval recipe)
        pl["pooled"][k] = pl["feats_final"][k].mean(0)
        del h, pixel
        torch.cuda.empty_cache()

    for k in tqdm(heroes, desc=f"metrics[{label}]"):
        i = keys.index(k)
        pixel = to_pixel(batch[i:i + 1])
        hero = {}

        # Scene B — future-frame MSE (m12d random-block mask; SAME sample across models)
        m_enc, m_pred = shared_masks[k]
        l1_b, out_t, tgt_t = masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred)
        err = (out_t - tgt_t).abs().mean(-1)[0].float().cpu().numpy()
        pred_idx = m_pred[0].cpu().numpy()
        hero["B_err"] = tokens_to_grid(err, pred_idx, num_frames)
        hid = np.zeros(n_tok, bool); hid[pred_idx] = True
        hero["B_hidden"] = hid.reshape(Tp, Hp, Wp)
        hero["B_l1"] = float(l1_b[0])

        # Scene C — causal future-block L1 (EXACT pt_causal mask)
        half = Tp // 2
        m_enc_c = expand_mask(temporal_token_idx(num_frames, range(0, half)), 1)
        m_pred_c = expand_mask(temporal_token_idx(num_frames, range(half, Tp)), 1)
        l1_c, out_c, tgt_c = masked_predict_l1(encoder, predictor, pixel, m_enc_c, m_pred_c)
        err_c = (out_c - tgt_c).abs().mean(-1)[0].float().cpu().numpy()
        idx_c = m_pred_c[0].cpu().numpy()
        hero["C_err"] = tokens_to_grid(err_c, idx_c, num_frames)
        hid_c = np.zeros(n_tok, bool); hid_c[idx_c] = True
        hero["C_hidden"] = hid_c.reshape(Tp, Hp, Wp)
        hero["C_l1"] = float(l1_c[0])
        hero["C_half"] = half
        hero["C_curve"] = np.nanmean(hero["C_err"][half:].reshape(Tp - half, -1), axis=1)
        causal_ref = pt_causal.compute(encoder, predictor, batch[i:i + 1], num_frames)
        if not np.allclose(causal_ref[0], l1_c[0], rtol=0.05):
            sys.exit(f"FATAL[{label}]: demo causal L1 {l1_c[0]:.5f} != pt_causal {causal_ref[0]:.5f}")

        # Scene D — mask-ratio robustness (EXACT pt_maskratio partition)
        g = torch.Generator().manual_seed(PT_SEED)
        perm = torch.randperm(n_tok, generator=g)
        ratio_l1, ratio_hidden = [], []
        for r in PT_MASK_RATIOS:
            kk = max(1, int(round(n_tok * r)))
            pred_i = perm[:kk].sort().values
            enc_i = perm[kk:].sort().values
            l1_r, _, _ = masked_predict_l1(
                encoder, predictor, pixel, expand_mask(enc_i, 1), expand_mask(pred_i, 1))
            ratio_l1.append(float(l1_r[0]))
            h2 = np.zeros(n_tok, bool); h2[pred_i.numpy()] = True
            ratio_hidden.append(h2.reshape(Tp, Hp, Wp))
        slope = float(perclip_slope(np.array(ratio_l1)[None, :], PT_MASK_RATIOS)[0])
        slope_ref = pt_maskratio.compute(encoder, predictor, batch[i:i + 1], num_frames)
        if not np.allclose(slope_ref[0], slope, rtol=0.05, atol=1e-4):
            sys.exit(f"FATAL[{label}]: demo maskratio slope {slope:.5f} != pt_maskratio {slope_ref[0]:.5f}")
        hero["D_l1"] = ratio_l1
        hero["D_hidden"] = ratio_hidden
        hero["D_slope"] = slope

        pl["hero"][k] = hero
        del pixel
        torch.cuda.empty_cache()

    del encoder, predictor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[m14] {label} done · GPU freed ({torch.cuda.memory_allocated() // 2**20} MiB left)")
    return pl


# ── Frame writer (png sequence → ffmpeg mp4) ───────────────────────────────────

class FrameWriter:
    def __init__(self, frames_dir: Path, dpi: int):
        self.dir, self.dpi, self.i = frames_dir, dpi, 0
        frames_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, fig, hold_frames: int):
        fig.savefig(self.dir / f"{self.i:05d}.png", dpi=self.dpi, facecolor=fig.get_facecolor())
        first = self.dir / f"{self.i:05d}.png"
        self.i += 1
        for _ in range(hold_frames - 1):
            (self.dir / f"{self.i:05d}.png").write_bytes(first.read_bytes())
            self.i += 1
        plt.close(fig)


def new_fig():
    return plt.figure(figsize=(12.8, 7.2), facecolor="#101418")


def _txt(fig, x, y, s, size, color="#ECEFF1", ha="center", weight="bold"):
    fig.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
             fontweight=weight, family="DejaVu Sans")


def _dark_axes(ax):
    ax.set_facecolor("#181C20")
    ax.tick_params(colors="#ECEFF1", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#455A64")


def _masked_frame(disp_fr, hidden2d, crop):
    m = upsample_grid(hidden2d.astype(np.float32), crop)[..., None]
    return (disp_fr.astype(np.float32) * (1.0 - 0.85 * m)).astype(np.uint8)


def _heat_overlay(ax, err2d, crop, alpha, vmax):
    hm = upsample_grid(err2d, crop)
    ax.imshow(np.ma.masked_invalid(hm), cmap=_HEAT_CMAP, alpha=alpha, vmin=0, vmax=vmax)


def _pretty(key):
    return key.split("_0")[0].replace("_", " · ")


# ── Scenes (generic over 1..2 models) ───────────────────────────────────────────

def scene_title(fw, hold, n_clips, labels):
    fig = new_fig()
    _txt(fig, 0.5, 0.72, "How the 4 headline metrics SEE a video", 30)
    vs = "   vs   ".join(labels)
    _txt(fig, 0.5, 0.58, vs, 20, "#80CBC4")
    _txt(fig, 0.5, 0.48, "V-JEPA 2.1 ViT-g (1B) encoder + its own predictor · "
                         f"{n_clips} real WalkIndia street clips (walking Goa · driving Delhi)", 13, "#B0BEC5")
    _txt(fig, 0.5, 0.32, "A · what the model sees      B · future-frame MSE      C · causal future-block L1", 13, "#90CAF9")
    _txt(fig, 0.5, 0.26, "D · mask-ratio robustness slope      E · motion-cosine separation", 13, "#90CAF9")
    _txt(fig, 0.5, 0.12, "every computation = the paper eval suite's own code (utils.pt_* / predictor_eval)", 11, "#607D8B")
    fw.emit(fig, hold)


def scene_features(fw, models, clips_disp, heroes, num_frames, dcfg, hold_step):
    """Scene A — [real frame | PCA per model], one row per hero clip, animated over slots."""
    Tp, _, _, _ = token_grid(num_frames)
    n_m = len(models)
    pcas = {(m["label"], k): pca_rgb(m["feats_final"][k], num_frames, dcfg["pca_components"])
            for m in models for k in heroes}
    ncol = 1 + n_m
    w = 0.86 / (ncol * len(heroes)) if len(heroes) > 1 else 0.80 / ncol
    for t in range(Tp):
        fig = new_fig()
        _txt(fig, 0.5, 0.95, "Scene A — what the model sees (dense features, PCA → RGB)", 19)
        _txt(fig, 0.5, 0.905, "same colour = tracked as the same thing across frames · "
                              f"time slot {t + 1}/{Tp}", 12, "#B0BEC5")
        for ci, k in enumerate(heroes):
            fr = clips_disp[k][min(t * 2, clips_disp[k].shape[0] - 1)]
            x0 = 0.05 + ci * (0.86 / len(heroes) + 0.03)
            axs = [fig.add_axes([x0 + j * (w + 0.008), 0.14, w, 0.60]) for j in range(ncol)]
            axs[0].imshow(fr)
            axs[0].set_title(_pretty(k), fontsize=11, color="#ECEFF1", fontweight="bold")
            for j, m in enumerate(models):
                axs[1 + j].imshow(pcas[(m["label"], k)][t])
                axs[1 + j].set_title(m["label"], fontsize=10, color=_MODEL_COLORS[j], fontweight="bold")
            for ax in axs:
                ax.set_xticks([]); ax.set_yticks([])
        fw.emit(fig, hold_step)


def scene_masked(fw, scene_id, title, subtitle, caption, models, k, disp, err_key, hidden_key,
                 val_key, crop, dcfg, hold_step, num_frames, curve=False):
    """Scenes B & C — [masked input | error per model] + value/curve panel."""
    n_m = len(models)
    Tp = models[0]["hero"][k][err_key].shape[0]
    vmax = max(np.nanmax(m["hero"][k][err_key]) for m in models)
    for t in range(Tp):
        fig = new_fig()
        _txt(fig, 0.5, 0.95, title, 19)
        _txt(fig, 0.5, 0.905, subtitle, 12, "#B0BEC5")
        fr = disp[min(t * 2, disp.shape[0] - 1)]
        w = 0.21
        ax1 = fig.add_axes([0.04, 0.16, w, 0.56])
        ax1.imshow(_masked_frame(fr, models[0]["hero"][k][hidden_key][t], crop))
        ax1.set_title(f"model input · slot {t + 1}/{Tp}\n(black = hidden)", fontsize=10,
                      color="#ECEFF1", fontweight="bold")
        ax1.set_xticks([]); ax1.set_yticks([])
        for j, m in enumerate(models):
            ax = fig.add_axes([0.04 + (j + 1) * (w + 0.015), 0.16, w, 0.56])
            ax.imshow(fr)
            _heat_overlay(ax, m["hero"][k][err_key][t], crop, dcfg["heat_alpha"], vmax)
            ax.set_title(f"{m['label']}\nerror = {m['hero'][k][val_key]:.4f}", fontsize=10,
                         color=_MODEL_COLORS[j], fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        axr = fig.add_axes([0.04 + (n_m + 1) * (w + 0.015) + 0.015, 0.20, 0.94 - (0.04 + (n_m + 1) * (w + 0.015) + 0.015), 0.48])
        _dark_axes(axr)
        if curve:
            half = models[0]["hero"][k]["C_half"]
            xs = np.arange(half + 1, Tp + 1)
            ymax = max(max(m["hero"][k]["C_curve"]) for m in models)
            upto = max(0, t + 1 - half)
            for j, m in enumerate(models):
                cv = m["hero"][k]["C_curve"]
                if upto > 0:
                    axr.plot(xs[:upto], cv[:upto], "o-", color=_MODEL_COLORS[j], lw=2,
                             label=m["label"])
            axr.set_xlim(half + 0.5, Tp + 0.5)
            axr.set_ylim(0, ymax * 1.25)
            axr.set_xlabel("future time slot", color="#ECEFF1", fontsize=9, fontweight="bold")
            axr.set_ylabel("prediction error (L1)", color="#ECEFF1", fontsize=9, fontweight="bold")
            if upto > 0:                       # legend only once a line exists (no-artist warning otherwise)
                axr.legend(fontsize=8, facecolor="#181C20", labelcolor="#ECEFF1", edgecolor="#455A64")
        else:
            axr.axis("off")
            best = min(models, key=lambda m: m["hero"][k][val_key])
            for j, m in enumerate(models):
                _txt(fig, 0.87, 0.56 - j * 0.14, f"{m['hero'][k][val_key]:.4f}", 22, _MODEL_COLORS[j])
                _txt(fig, 0.87, 0.50 - j * 0.14, m["label"], 10, _MODEL_COLORS[j])
            if n_m > 1:
                _txt(fig, 0.87, 0.27, f"lower = better\n→ {best['label']} wins", 11, "#80CBC4")
            else:
                _txt(fig, 0.87, 0.27, "mean latent L1\n(lower = better)", 11, "#B0BEC5")
        _txt(fig, 0.5, 0.055, caption, 12, "#80CBC4")
        fw.emit(fig, hold_step)


def scene_maskratio(fw, models, k, disp, crop, dcfg, hold_step, num_frames):
    """Scene D — masked mosaic + one L1-vs-r line per model; the slope IS the metric."""
    ratios = list(PT_MASK_RATIOS)
    Tp, _, _, _ = token_grid(num_frames)
    mid = Tp // 2
    all_l1 = [v for m in models for v in m["hero"][k]["D_l1"]]
    for j_r, r in enumerate(ratios):
        fig = new_fig()
        _txt(fig, 0.5, 0.95, "Scene D — mask-ratio robustness slope", 19)
        _txt(fig, 0.5, 0.905, f"hide {int(r * 100)}% of ALL spacetime patches — same fixed shuffle as the "
                              f"eval (seed {PT_SEED}) · clip: {_pretty(k)}", 12, "#B0BEC5")
        fr = disp[min(mid * 2, disp.shape[0] - 1)]
        ax1 = fig.add_axes([0.05, 0.16, 0.26, 0.56])
        ax1.imshow(_masked_frame(fr, models[0]["hero"][k]["D_hidden"][j_r][mid], crop))
        ax1.set_title(f"model input ({int(r * 100)}% hidden)", fontsize=11, color="#ECEFF1",
                      fontweight="bold")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax3 = fig.add_axes([0.42, 0.18, 0.52, 0.54])
        _dark_axes(ax3)
        for j, m in enumerate(models):
            l1 = m["hero"][k]["D_l1"]
            ax3.plot(ratios[:j_r + 1], l1[:j_r + 1], "o-", color=_MODEL_COLORS[j], lw=2.5, ms=8,
                     label=f"{m['label']}  (slope {m['hero'][k]['D_slope']:.4f})")
            if j_r == len(ratios) - 1:
                ax3.plot(ratios, np.poly1d(np.polyfit(ratios, l1, 1))(ratios), "--",
                         color=_MODEL_COLORS[j], lw=1.5, alpha=0.7)
        ax3.set_xlim(ratios[0] - 0.07, ratios[-1] + 0.07)
        ax3.set_ylim(min(all_l1) * 0.97, max(all_l1) * 1.03)
        ax3.set_xlabel("fraction of video hidden", color="#ECEFF1", fontsize=10, fontweight="bold")
        ax3.set_ylabel("latent L1 error", color="#ECEFF1", fontsize=10, fontweight="bold")
        ax3.legend(fontsize=9, facecolor="#181C20", labelcolor="#ECEFF1", edgecolor="#455A64",
                   loc="upper left")
        if j_r == len(ratios) - 1:
            best = min(models, key=lambda m: m["hero"][k]["D_slope"])
            ax3.set_title(f"the SLOPE of each line IS the metric — flatter = degrades more "
                          f"gracefully{(' → ' + best['label'] + ' wins') if len(models) > 1 else ''}",
                          fontsize=11, color="#FFB74D", fontweight="bold")
        _txt(fig, 0.5, 0.055, "a jigsaw with more and more missing pieces — how gracefully does each "
                              "model degrade? (lower slope = better)", 12, "#80CBC4")
        fw.emit(fig, hold_step)


def scene_motion_cos(fw, models, clips_disp, keys, hold):
    """Scene E — one cosine matrix per model + margins."""
    n_m = len(models)
    types = [k.split("_")[0] for k in keys]
    short = [_pretty(k).replace(" · ", "\n") for k in keys]
    results = {}
    fig = new_fig()
    _txt(fig, 0.5, 0.95, "Scene E — motion-cosine separation", 19)
    _txt(fig, 0.5, 0.905, "do same-motion clips LOOK ALIKE to the model? (pooled encoder features, "
                          "cosine) · the GAP same−cross is the metric", 12, "#B0BEC5")
    for j, m in enumerate(models):
        F = np.stack([m["pooled"][k] for k in keys])
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        C = F @ F.T
        same = [C[a, b] for a in range(len(keys)) for b in range(len(keys)) if a < b and types[a] == types[b]]
        cross = [C[a, b] for a in range(len(keys)) for b in range(len(keys)) if a < b and types[a] != types[b]]
        margin = float(np.mean(same) - np.mean(cross))
        results[m["label"]] = {"margin": margin, "matrix": C.tolist()}
        axm = fig.add_axes([0.07 + j * 0.46, 0.16, 0.34, 0.56])
        axm.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        for a in range(len(keys)):
            for b in range(len(keys)):
                axm.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=11,
                         color="black", fontweight="bold")
        axm.set_xticks(range(len(keys))); axm.set_yticks(range(len(keys)))
        axm.set_xticklabels(short, fontsize=8, color="#ECEFF1", fontweight="bold")
        axm.set_yticklabels(short, fontsize=8, color="#ECEFF1", fontweight="bold")
        axm.set_title(f"{m['label']}\nseparation margin = {margin:+.3f}", fontsize=12,
                      color=_MODEL_COLORS[j], fontweight="bold")
    if n_m > 1:
        best = max(results, key=lambda L: results[L]["margin"])
        _txt(fig, 0.5, 0.065, f"higher margin = same-motion clips cluster tighter → {best} wins", 12, "#80CBC4")
    else:
        _txt(fig, 0.5, 0.065, "walking↔walking and driving↔driving should beat walking↔driving", 12, "#80CBC4")
    fw.emit(fig, hold)
    return results


def scene_verdict(fw, models, heroes, cos_results, hold):
    fig = new_fig()
    n_m = len(models)
    _txt(fig, 0.5, 0.93, "Verdict — the 4 metrics on these clips (all ↓ lower better, except margin ↑)", 19)
    metrics = [("future-frame MSE ↓", "B_l1"), ("causal future-block L1 ↓", "C_l1"),
               ("mask-ratio slope ↓", "D_slope")]
    y = 0.82
    for k in heroes:
        _txt(fig, 0.10, y, _pretty(k), 13, "#90CAF9", ha="left")
        y -= 0.055
        for name, mk in metrics:
            _txt(fig, 0.14, y, name, 11, "#B0BEC5", ha="left")
            vals = [m["hero"][k][mk] for m in models]
            best = int(np.argmin(vals))
            for j, m in enumerate(models):
                col = "#80CBC4" if (n_m > 1 and j == best) else _MODEL_COLORS[j]
                mark = "  ◀ wins" if (n_m > 1 and j == best) else ""
                _txt(fig, 0.52 + j * 0.24, y, f"{vals[j]:.4f}{mark}", 11, col, ha="left")
            y -= 0.048
        y -= 0.02
    _txt(fig, 0.14, y, "motion-cosine separation margin ↑ (all clips)", 11, "#B0BEC5", ha="left")
    margins = [cos_results[m["label"]]["margin"] for m in models]
    bestm = int(np.argmax(margins))
    for j, m in enumerate(models):
        col = "#80CBC4" if (n_m > 1 and j == bestm) else _MODEL_COLORS[j]
        mark = "  ◀ wins" if (n_m > 1 and j == bestm) else ""
        _txt(fig, 0.52 + j * 0.24, y, f"{margins[j]:+.3f}{mark}", 11, col, ha="left")
    for j, m in enumerate(models):
        _txt(fig, 0.52 + j * 0.24, 0.88, m["label"], 12, _MODEL_COLORS[j], ha="left")
    _txt(fig, 0.5, 0.10, "the paper's eval scores these SAME 4 computations on 23,106 held-out clips —", 11, "#B0BEC5")
    _txt(fig, 0.5, 0.06, "there, surgery separates from the best competitor by 43.3x / 33.2x / 20.0x / 13.9x CI", 11, "#80CBC4")
    fw.emit(fig, hold)


# ── Contact sheet (for the visual-audit agent) ──────────────────────────────────

def contact_sheet(frames_dir: Path, out_png: Path, tiles: int):
    from PIL import Image
    pngs = sorted(frames_dir.glob("*.png"))
    idx = np.linspace(0, len(pngs) - 1, tiles).round().astype(int)
    ims = [Image.open(pngs[i]).resize((480, 270)) for i in idx]
    side = int(np.ceil(np.sqrt(tiles)))
    sheet = Image.new("RGB", (side * 480, side * 270), "black")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % side) * 480, (k // side) * 270))
    sheet.save(out_png)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="m14 — visual demo of the 4 headline metrics (A/B capable)")
    p.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                   help="repeatable: 'FROZEN 2.1=checkpoints/….pt' (1 = single demo, 2 = A/B)")
    p.add_argument("--model-config", type=Path, required=True, help="configs/model/*.yaml")
    p.add_argument("--demo-config", type=Path, required=True, help="configs/demo.yaml")
    p.add_argument("--clips-dir", type=Path, required=True,
                   help="dir of <tourtype>_<city>_<shard>_<key>.mp4 (+ .json) demo clips")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("FATAL: CUDA required (1B encoder+predictor forwards) — no CPU fallback")
    specs = []
    for spec in args.ckpt:
        if "=" not in spec:
            sys.exit(f"FATAL: --ckpt must be LABEL=PATH, got '{spec}'")
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            sys.exit(f"FATAL: ckpt not found: {path}")
        specs.append((label, Path(path)))
    if len(specs) > 2:
        sys.exit("FATAL: at most 2 --ckpt (A/B) supported")

    dcfg = yaml.safe_load(args.demo_config.read_text())["demo"]
    num_frames = get_pipeline_config()["probe"]["num_frames"]      # metric parity with m12d/m12e
    crop = yaml.safe_load(args.model_config.read_text())["model"]["crop_size"]
    fps = dcfg["fps"]
    hold_step = max(1, round(dcfg["hold_step_s"] * fps))
    hold_title = max(1, round(dcfg["hold_title_s"] * fps))

    out = args.output_dir
    frames_dir = out / f"frames_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(args.clips_dir.glob("*.mp4"))
    if not mp4s:
        sys.exit(f"FATAL: no .mp4 in {args.clips_dir}")
    print(f"[m14] {len(mp4s)} demo clips · num_frames={num_frames} · crop={crop} · "
          f"models: {[s[0] for s in specs]}")

    # ── decode + preprocess (exact eval recipe) ──
    clips_disp, batch_rows, keys = {}, [], []
    for mp4 in tqdm(mp4s, desc="decode"):
        fr = decode_uniform_frames(mp4, num_frames)
        norm = resize_and_normalize(fr, crop)
        clips_disp[mp4.stem] = denorm_to_uint8(norm)
        batch_rows.append(norm)
        keys.append(mp4.stem)
    batch = torch.stack(batch_rows)

    # heroes: first clip of each tour_type
    heroes = []
    for k in keys:
        if k.split("_")[0] not in [h.split("_")[0] for h in heroes]:
            heroes.append(k)
    heroes = heroes[: dcfg["n_hero_clips"]]
    print(f"[m14] hero clips: {heroes}")

    # shared scene-B masks: ONE _MaskGenerator sample per hero clip, reused across models
    mask_gen = build_mask_gen(num_frames)
    shared_masks = {}
    for k in heroes:
        me_raw, mp_raw = mask_gen(1)
        m_enc = (torch.stack(me_raw, 0) if isinstance(me_raw, list) else me_raw).to("cuda")
        m_pred = (torch.stack(mp_raw, 0) if isinstance(mp_raw, list) else mp_raw).to("cuda")
        shared_masks[k] = (m_enc, m_pred)

    # ── per-model payloads (SEQUENTIAL — one model on the GPU at a time) ──
    models = [compute_payload(label, ck, args.model_config, batch, keys, heroes, num_frames,
                              shared_masks) for label, ck in specs]

    # ── render ──
    fw = FrameWriter(frames_dir, dcfg["dpi"])
    scene_title(fw, hold_title, len(keys), [m["label"] for m in models])
    scene_features(fw, models, clips_disp, heroes, num_frames, dcfg, hold_step)
    for k in heroes:
        Tp, _, _, _ = token_grid(num_frames)
        half = Tp // 2
        scene_masked(fw, "B", f"Scene B — future-frame MSE   ·   {_pretty(k)}",
                     "random spacetime blocks hidden (the m12d eval mask, SAME blocks for every model) — "
                     "predict them in FEATURE space",
                     "cover parts of the flip-book and ask each model to draw them — bright = drawn wrong",
                     models, k, clips_disp[k], "B_err", "B_hidden", "B_l1", crop, dcfg, hold_step,
                     num_frames)
        scene_masked(fw, "C", f"Scene C — causal future-block L1   ·   {_pretty(k)}",
                     f"the ENTIRE second half (slots {half + 1}–{Tp}) is hidden — predict the future from "
                     "the past only (the pt_causal eval mask)",
                     "hide the second half of the movie — who guesses it better from the first half?",
                     models, k, clips_disp[k], "C_err", "C_hidden", "C_l1", crop, dcfg, hold_step,
                     num_frames, curve=True)
        scene_maskratio(fw, models, k, clips_disp[k], crop, dcfg, hold_step, num_frames)
    cos_results = scene_motion_cos(fw, models, clips_disp, keys, hold_title)
    scene_verdict(fw, models, heroes, cos_results, hold_title)

    # ── assemble mp4 + contact sheet + metrics json ──
    mp4_path = out / "demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", str(dcfg["crf"]), str(mp4_path)], check=True)
    contact_sheet(frames_dir, out / "contact_sheet.png", dcfg["sheet_tiles"])
    mj = {"num_frames": num_frames, "mask_ratios": list(PT_MASK_RATIOS), "models": {}}
    for m in models:
        mj["models"][m["label"]] = {
            "per_clip": {k: {kk: vv for kk, vv in m["hero"][k].items()
                             if kk in ("B_l1", "C_l1", "D_slope", "D_l1")}
                         for k in heroes},
            "motion_cos": cos_results[m["label"]],
        }
    (out / "demo_metrics.json").write_text(json.dumps(mj, indent=1))
    print(f"[m14] DONE → {mp4_path} ({fw.i} frames @ {fps} fps = {fw.i / fps:.0f}s) · "
          f"contact_sheet.png · demo_metrics.json")


if __name__ == "__main__":
    main()
