"""V-JEPA 2.1 HEAD-ONLY surgery: factor-aug data + FROZEN encoder + FROZEN predictor. GPU-only.
Gold standard: https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py
Claude Code: re-WebSearch this URL on every read of this file.

iter15 Phase 2 (2026-05-14): head-only sibling of m09c1_surgery_encoder.py. All 48 ViT-G
blocks + predictor FROZEN; only the motion_aux head trains. Differs from m09a2 by the
DATA path — uses StreamingFactorDataset for D_L/D_A/D_I factor-augmented clips per the
recipe-v3 mode_mixture. Single training stage (vs m09c1's 2-3 progressive unfreeze stages)
since the encoder is frozen always. Rule 32: zero cross-imports from m09a1/m09a2/m09c1;
shared primitives via utils/training.py + utils/motion_aux_loss.py + utils/factor_streaming.py.

USAGE — FULL arg set (run_train.sh is the canonical caller; it wires EVERY arg below.
Eyeball to confirm nothing is silently cfg-defaulted. <LD> = pipeline.yaml
data.local_data_dir; <M> = sanity|poc|full; swap --FULL→--SANITY/--POC for other tiers):
    python -u src/m09c2_surgery_head.py --FULL \
        --model-config         configs/model/vjepa2_1.yaml \
        --train-config         configs/train/surgery_3stage_DI_head.yaml \
        --subset               <LD>/train_pool.json \
        --local-data           <LD> \
        --val-subset           <LD>/val_split.json \
        --val-local-data       <LD> \
        --init-from-ckpt       <hf://owner/repo/file OR local .pt — run_train.sh $SURGERY_INIT> \
        --probe-subset         outputs/<M>/probe_action/action_labels.json \
        --probe-local-data     <LD> \
        --probe-tags           <LD>/tags.json \
        --probe-action-labels  outputs/<M>/probe_action/action_labels.json \
        --motion-features-path <LD>/m04d_motion_features/motion_features.npy \
        --taxonomy-labels-json outputs/<M>/probe_taxonomy/taxonomy_labels.json \
        --output-dir           outputs/<M>/m09c_surgery_3stage_DI_head \
        --cache-policy         <1=keep|2=recompute> \
        --no-wandb 2>&1 | tee logs/m09c2_surgery_3stage_DI_head_full.log
        
    # HEAD-ONLY surgery: encoder+predictor FROZEN. --subset = leakage-safe pool
    #   (clip_splits, iter17); _build_factor_loader restricts the streaming universe to it
    #   (test excluded). factor_manifest + masks derived from --local-data via data_paths
    #   (NOT a --factor-dir CLI arg today — promotion candidate for c1/c2 symmetry).
    #   noDI variant: --train-config surgery_2stage_noDI_head.yaml.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils.live_debug import install_debug_handlers
install_debug_handlers()

from utils.config import (
    check_gpu, get_module_output_dir, load_subset,
    get_pipeline_config, load_merged_config,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.progress import make_pbar
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.cache_policy import resolve_cache_policy_interactive, wipe_output_dir
from utils.m09_common import add_m09_common_args, merge_m09_common_config
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
)
from utils import data_paths   # iter17: canonical local-data path accessors (single source)
from utils.training import (
    load_config,
    build_optimizer, build_scheduler,
    assert_encoder_frozen, set_trainable_prefix,
    export_student_for_eval,
    compute_val_motion_aux_loss,   # iter17 DRY: shared with m09a2 (was local _compute_…)
    StreamingFactorDataset, build_streaming_indices, _streaming_worker_init,
    # iter15 Phase 6 C1+C2 (2026-05-16): trio + head-drift wiring for head cells.
    build_probe_clips, run_trio_at_val, track_head_drift_at_val,
    build_mask_generators,
)
from utils.action_labels import load_action_labels
from utils.motion_aux_loss import (
    build_motion_aux_head_from_cfg,
    attach_motion_aux_to_optimizer, run_motion_aux_step,
    export_motion_aux_head,
)
from utils.probe_labels import ensure_probe_labels_for_mode
# iter15 Phase 6 audit (2026-05-16): emit-parity with m09c1 — same loss_log
# schema + plot calls so outputs/{poc,full}/m09c_surgery_*_head/ matches the
# encoder-cell layout.
from utils.plots import (
    plot_training_curves, plot_combined_losses,
    plot_probe_trajectory_trio, plot_val_loss_with_kill_switch_overlay,
)

CHECKPOINT_PREFIX = "m09c_ckpt"  # output filename preserved for downstream eval compat
_pcfg = get_pipeline_config()


def merge_config_with_args(cfg: dict, args) -> dict:
    """Mode-gated config merge: delegates to utils.m09_common (shared with m09a/c)."""
    if args.SANITY:
        mode_key = "sanity"
    elif args.POC:
        mode_key = "poc"
    else:
        mode_key = "full"
    merge_m09_common_config(cfg, args, mode_key)

    # iter15 (2026-05-17): mirror m09c1:159 — pass --init-from-ckpt through to cfg
    # so build_model can read it. Required for Δ5 paired-test validity: both
    # surg_DI_enc (m09c1) and surg_DI_head (m09c2) MUST start from the same
    # post-pretrain init (was previously Meta baseline → invalidated Δ5).
    cfg["init_from_ckpt"] = args.init_from_ckpt

    # === Head-only contract: ENFORCED at runtime, DECLARED in yaml — not hardcoded here ===
    # Head-only encoder freeze is enforced in build_model() via:
    #   1. set_trainable_prefix(student, 0)  → all blocks frozen
    #   2. for p in (encoder + predictor): requires_grad = False
    #   3. assert_encoder_frozen(student)    → fail-loud guard
    # + train()'s "n_enc_pred_params == 0" optimizer guard (fail loud). The loss-weight /
    # drift values (loss.weight_jepa=0.0, loss.weight_motion_aux=1.0, drift_control.enabled
    # =false) are DECLARED in configs/train/surgery_*_head.yaml — the single source. We do
    # NOT re-set or assert them against literals (CLAUDE.md: no hardcoded values, no
    # fallback; an assert-vs-literal just relocates the hardcode). Nothing in m09c2 reads
    # them, and a missing yaml key raises KeyError = fail loud. The per-stage validation
    # below requires stages[0].unfreeze_below=0.0 (read from yaml, not forced).

    # === Force single head-only stage ===
    # The yaml SHOULD already declare exactly one stage with unfreeze_below=0.0
    # (per pretrain_head + surgery_*_head.yaml). Fail loud if it doesn't.
    stages = cfg["surgery"]["stages"]
    if len(stages) != 1:
        print(f"FATAL [m09c2]: head-only mode requires exactly 1 surgery stage in yaml; "
              f"found {len(stages)}. Use configs/train/surgery_*_head.yaml not _encoder.yaml.")
        sys.exit(1)
    if stages[0]["unfreeze_below"] != 0.0:
        print(f"FATAL [m09c2]: head-only stage 0 must set unfreeze_below=0.0; "
              f"got {stages[0]['unfreeze_below']}.")
        sys.exit(1)

    # === Factor streaming: read mode-gated flag from yaml (mirror m09c1:225-231) ===
    # Head-only surgery has ONLY a streaming data path (StreamingFactorDataset; no legacy
    # FactorSampler). Streaming is therefore MANDATORY — surgery_*_head.yaml declares
    # factor_streaming.{sanity,poc,full}=true (overriding surgery_base's sanity:false).
    # We READ the flag (single source) instead of forcing it (CLAUDE.md no-hardcode), and
    # FAIL LOUD if a yaml ever sets it false: m09c2 physically cannot honor that.
    fs_cfg = cfg["factor_streaming"]
    fs_enabled = fs_cfg[mode_key]
    if not fs_enabled:
        print(f"FATAL [m09c2]: head-only surgery is streaming-ONLY; factor_streaming."
              f"{mode_key}=false cannot be honored. Set it true in the head yaml "
              f"(surgery_*_head.yaml declares factor_streaming.sanity=true).")
        sys.exit(1)
    cfg["factor_streaming"]["enabled"] = fs_enabled
    cfg["factor_streaming"]["num_workers"] = fs_cfg["num_workers"][mode_key]

    # === Output dir: explicit --output-dir, or auto from mode ===
    if args.output_dir is not None:
        cfg["checkpoint"]["output_dir"] = args.output_dir
    else:
        base_out = get_module_output_dir(
            "m09c2_surgery_head", args.subset,
            sanity=args.SANITY, poc=args.POC)
        # Append variant tag so 3stage_DI_head + noDI_head write to DIFFERENT subdirs
        # (downstream eval scans by encoder name). variant_tag is DECLARED in the head
        # yaml's data block (single source) — was a hardcoded .replace("vjepa_2_1_surgical_",
        # "") string-strip (CLAUDE.md no-hardcode). Missing key → KeyError = fail loud.
        variant_tag = cfg["data"]["variant_tag"]
        cfg["checkpoint"]["output_dir"] = str(base_out / variant_tag)
    return cfg


def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder + predictor, BOTH FROZEN. assert_encoder_frozen() validates."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    arch = model_cfg["arch"]

    vit_constructor = get_vit_by_arch(arch)
    vit_predictor = get_vit_predictor()

    crop_size = model_cfg["crop_size"]
    student = vit_constructor(
        img_size=(crop_size, crop_size),
        patch_size=model_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=model_cfg["tubelet_size"],
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=model_cfg["use_rope"],
        use_activation_checkpointing=model_cfg["use_activation_checkpointing"],
    )

    # iter15 (2026-05-17): mirror m09c1:319-352 init_from_ckpt dispatcher.
    # Accepts BOTH hf:// URIs AND local filesystem paths for ANY mode. Replaces
    # the previous Meta V-JEPA 2.1 baseline load (which produced an invalid Δ5
    # paired test — m09c1 starts from post-pretrain, m09c2 was starting from
    # Meta baseline). Schema: prior-run ckpt MUST carry "student" + "predictor"
    # keys (same contract as m09c1 — verified at hf://anonymousML123/
    # factorjepa-pretrain-vjepa21-vitg-5ep/m09a_ckpt_best.pt).
    project_root = Path(__file__).parent.parent
    init_from = cfg["init_from_ckpt"]   # always set — argparse required=True

    if init_from.startswith("hf://"):
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download
        load_dotenv(project_root / ".env")
        uri = init_from[len("hf://"):]
        parts = uri.split("/", 2)
        if len(parts) < 3:
            print(f"FATAL: bad --init-from-ckpt URI: {init_from}")
            print("  Expected: hf://<owner>/<repo>/<filename>")
            sys.exit(1)
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("FATAL: HF_TOKEN missing in .env — required for HF model-repo download.")
            print(f"  Repo: {repo_id}")
            print("  Fix: add HF_TOKEN=hf_... to .env (project root)")
            sys.exit(1)
        print(f"  [iter15] HF download: {repo_id}/{filename}")
        ckpt_path = Path(hf_hub_download(
            repo_id=repo_id, filename=filename, token=hf_token))
        print(f"  [iter15] HF cached at: {ckpt_path}")
    else:
        ckpt_path = Path(init_from)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if not ckpt_path.is_file():
            print(f"FATAL: --init-from-ckpt local path not found: {ckpt_path}")
            print(f"  Resolved from: {init_from}")
            sys.exit(1)
        print(f"  [iter15] Local init from: {ckpt_path}")

    print(f"Loading pretrained weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # FAIL LOUD on schema mismatch (mirrors m09c1:360-369): prior-run ckpt MUST
    # carry "student" + "predictor" keys. NOT the Meta baseline schema
    # ("target_encoder"/"encoder") — that path is gone with iter15 parity fix.
    if not (isinstance(ckpt, dict) and "student" in ckpt
            and isinstance(ckpt["student"], dict)
            and "predictor" in ckpt
            and isinstance(ckpt["predictor"], dict)):
        print(f"FATAL: init ckpt missing 'student' + 'predictor' schema: {ckpt_path}")
        top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
        print(f"  Top-level: {top_keys}")
        print("  m09c2 accepts ONLY full prior-run ckpt schema (m09a_ckpt_best.pt /"
              " m09c_ckpt_best.pt). NOT Meta baseline (vjepa2_1_vitG_384.pt).")
        sys.exit(1)
    state_dict = ckpt["student"]
    print(f"  [iter15] Schema: student ({len(state_dict)} keys) + "
          f"predictor ({len(ckpt['predictor'])} keys)")
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in state_dict.items()}
    msg = student.load_state_dict(state_dict, strict=False)
    total_keys = len(list(student.state_dict().keys()))
    loaded_keys = total_keys - len(msg.missing_keys)
    load_pct = loaded_keys / max(total_keys, 1) * 100
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params "
          f"({loaded_keys}/{total_keys} keys = {load_pct:.0f}%)")
    if msg.missing_keys:
        unexpected_missing = [k for k in msg.missing_keys if "pos_embed" not in k]
        if unexpected_missing:
            print(f"FATAL: {len(unexpected_missing)} unexpected missing keys in student ckpt")
            for k in unexpected_missing[:10]:
                print(f"    {k}")
            sys.exit(1)
    if load_pct < model_cfg["min_student_load_pct"]:
        print(f"FATAL: only {load_pct:.0f}% of student keys loaded. Ckpt incompatible.")
        sys.exit(1)
    student = student.to(device)
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = True

    # === FREEZE encoder via set_trainable_prefix(0) — mirrors m09c1 stage logic ===
    # set_trainable_prefix(student, 0) sets all blocks frozen; norms remain trainable
    # by Meta convention. We immediately re-freeze norms here to ensure STRICT zero
    # encoder gradient (head-only contract).
    set_trainable_prefix(student, 0)
    for p in student.parameters():
        p.requires_grad = False
    assert_encoder_frozen(student)
    student.eval()
    print("[m09c2 STRICT HEAD-ONLY] encoder FROZEN: 0 trainable block params (asserted)")

    # === Predictor: load Meta weights, FROZEN ===
    pred_constructor = get_vit_predictor_2_1() if model_cfg["predict_all"] else vit_predictor
    predictor = pred_constructor(
        img_size=(crop_size, crop_size),
        patch_size=model_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=model_cfg["tubelet_size"],
        embed_dim=model_cfg["embed_dim"],
        predictor_embed_dim=model_cfg["pred_embed_dim"],
        depth=model_cfg["pred_depth"],
        num_heads=model_cfg["pred_num_heads"],
        use_mask_tokens=True,
        num_mask_tokens=model_cfg["num_mask_tokens"],
        zero_init_mask_tokens=True,
        use_rope=model_cfg["use_rope"],
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=model_cfg["use_activation_checkpointing"],
        return_all_tokens=model_cfg["predict_all"],
    )
    if "predictor" not in ckpt:
        print("FATAL: ckpt has no 'predictor' key — V-JEPA 2.1 distribution must include it")
        sys.exit(1)
    pred_state = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in ckpt["predictor"].items()}
    pred_msg = predictor.load_state_dict(pred_state, strict=False)
    pred_total = len(list(predictor.state_dict().keys()))
    pred_loaded = pred_total - len(pred_msg.missing_keys)
    pred_pct = pred_loaded / max(pred_total, 1) * 100
    print(f"Predictor loaded: {pred_loaded}/{pred_total} keys = {pred_pct:.0f}%")
    if pred_pct < model_cfg["min_predictor_load_pct"]:
        print(f"FATAL: predictor only {pred_pct:.0f}% loaded")
        sys.exit(1)
    for p in predictor.parameters():
        p.requires_grad = False
    predictor = predictor.to(device)
    predictor.eval()
    print(f"Predictor: {sum(p.numel() for p in predictor.parameters()):,} params (FROZEN)")

    init_ckpt_path = str(ckpt_path)

    del ckpt
    gc.collect()

    return {
        "student": student,
        "predictor": predictor,
        "init_ckpt_path": init_ckpt_path,
        "explora_enabled": False,
    }


def _build_factor_loader(cfg: dict, train_keys: list, mode_mixture: dict,
                         stage_steps: int, base_seed: int) -> DataLoader:
    """Construct StreamingFactorDataset + DataLoader for factor-aug clips.

    Mirrors m09c1's setup (src/m09c1_surgery_encoder.py:1237-1262) — the
    StreamingFactorDataset generates D_L / D_A / D_I tubes on demand from
    (raw_mp4, mask.npz) pairs per the recipe-v3 mode_mixture.
    """
    data_cfg = cfg["data"]
    num_frames = data_cfg["num_frames"]
    crop_size = cfg["model"]["crop_size"]
    batch_size = cfg["optimization"]["batch_size"]
    local_data = data_cfg["local_data"]
    # iter17 (2026-05-26): single-source via data_paths (was local_data + hardcoded
    # subpaths → divergent from m09c1's --factor-dir; CLAUDE.md SHARED DERIVATION VIA CLI).
    manifest_path = data_paths.factor_manifest_path(local_data)
    masks_dir = data_paths.masks_dir(local_data)
    if not manifest_path.exists():
        print(f"FATAL: factor_manifest.json missing at {manifest_path}. "
              f"Run scripts/run_factor_prep.sh to generate.")
        sys.exit(1)
    # build_streaming_indices returns 3-tuple (m09c1:659 baseline):
    # (mp4_index, mask_index, streaming_manifest). The manifest re-read below is
    # redundant — fn already loads + validates it during the scan.
    mp4_index, mask_index, streaming_manifest = build_streaming_indices(
        manifest_path=manifest_path,
        masks_dir=masks_dir,
        local_data=local_data,
    )
    # iter17 (2026-05-26) — LEAKAGE FIX: restrict the streaming universe to train_keys
    # (= the leakage-safe --subset pool = corpus manifest − val − test, built once by
    # src/utils/clip_splits.py). m09c2 previously streamed the FULL factor manifest
    # regardless of train_keys → eval TEST clips leaked into the head-surgery pool.
    # Mirrors m09c1:676-677 (which already had this filter; m09c2 was missing it).
    # See CLAUDE.md "SHARED DERIVATION VIA CLI".
    _train_set = set(train_keys)
    n_before = len(mp4_index)
    mp4_index = {k: v for k, v in mp4_index.items() if k in _train_set}
    mask_index = {k: v for k, v in mask_index.items() if k in _train_set}
    print(f"  [m09c2 leakage-guard] streaming universe restricted to train pool: "
          f"{n_before} → {len(mp4_index)} clips (val∪test excluded upstream)", flush=True)
    if not mp4_index:
        print("FATAL [m09c2]: streaming universe EMPTY after train-pool filter — "
              "check --subset pool vs factor_manifest key overlap.", file=sys.stderr)
        sys.exit(1)
    # Build factor_cfg by remapping yaml keys → stream_factor's expected names.
    # Mirrors m09c1:667-673 baseline. FAIL LOUD per CLAUDE.md: cfg["factor_datasets"]
    # crashes here if the yaml chain is broken (no .get() fallback to silent {}).
    fcy = cfg["factor_datasets"]
    factor_cfg_streaming = {
        "layout_method": fcy["layout_patch_method"],
        "agent_method":  fcy["agent_patch_method"],
        "matte_factor":  fcy["soft_matte_factor"],
        "blur_sigma":    fcy["blur_sigma"],
        "feather_sigma": fcy["feather_sigma"],
    }
    replay_pct = cfg["replay"]["raw_pretrain_pct"]
    ds = StreamingFactorDataset(
        mp4_index=mp4_index,
        mask_index=mask_index,
        factor_manifest=streaming_manifest,
        factor_cfg=factor_cfg_streaming,
        mode_mixture=mode_mixture,
        num_frames=num_frames,
        crop_size=crop_size,
        di_legacy_index=None,
        base_seed=base_seed,
        steps_per_epoch=stage_steps * batch_size,
        interaction_cfg=cfg["interaction_mining"],
        raw_replay_pct=replay_pct,
        raw_clip_keys=None,
    )
    fs_cfg = cfg["factor_streaming"]
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=fs_cfg["num_workers"],
        prefetch_factor=fs_cfg["prefetch_factor"] if fs_cfg["num_workers"] > 0 else None,
        persistent_workers=fs_cfg["persistent_workers"] if fs_cfg["num_workers"] > 0 else False,
        pin_memory=fs_cfg["pin_memory"],
        worker_init_fn=_streaming_worker_init if fs_cfg["num_workers"] > 0 else None,
    )
    return loader


def train(cfg: dict, args) -> None:
    """Head-only surgery: frozen encoder + predictor; only motion_aux head trains.

    Data path: factor-augmented clips via StreamingFactorDataset (mode_mixture from
    yaml; e.g. 3stage_DI = {L:0.15, A:0.15, I:0.70}, noDI = {L:0.50, A:0.50, I:0.00}).
    """
    check_gpu()
    print_cgroup_header(prefix="[m09c2]")
    start_oom_watchdog(prefix="[m09c2]-oom-watchdog")
    device = torch.device("cuda")

    seed = cfg["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(cfg["checkpoint"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    train_keys = load_subset(args.subset)
    val_keys = load_subset(args.val_subset)
    # iter17 (2026-05-26): clip-count per mode is governed by clip_pool_ratio[mode] (pipeline.yaml)
    # applied in clip_splits.py → the --subset train_pool is already ratio-scaled (SANITY/POC/FULL),
    # which also shrinks _build_factor_loader's streaming universe. No fixed in-trainer cap.
    print(f"Train: {len(train_keys):,} keys · Val: {len(val_keys):,} keys")

    mode_key = "sanity" if args.SANITY else ("poc" if args.POC else "full")
    mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
    ensure_probe_labels_for_mode(
        mode_flag=mode_flag,
        project_root=Path(__file__).parent.parent,
        cache_policy=args.cache_policy,
        cfg=cfg,
    )

    # === Build model (frozen encoder + frozen predictor) ===
    model_d = build_model(cfg, device)
    student = model_d["student"]
    predictor = model_d["predictor"]
    init_ckpt_path = model_d["init_ckpt_path"]

    # === Build motion_aux head ===
    ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)
    if ma_head is None:
        print("FATAL [m09c2]: motion_aux head REQUIRED — sole training signal in head-only mode")
        sys.exit(1)

    # === Optimizer over head params only ===
    opt_cfg = cfg["optimization"]
    optimizer = build_optimizer(student, predictor, opt_cfg, init_params=None)
    n_enc_pred_params = sum(p.numel() for grp in optimizer.param_groups for p in grp["params"])
    if n_enc_pred_params > 0:
        print(f"FATAL [m09c2]: build_optimizer returned {n_enc_pred_params:,} encoder/predictor "
              f"trainable params — expected 0. Check requires_grad freeze in build_model().")
        sys.exit(1)
    attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, base_lr=opt_cfg["lr"])
    head_params = sum(p.numel() for p in ma_head.parameters() if p.requires_grad)
    print(f"Trainable params: motion_aux head = {head_params:,} (~432K expected)")

    # iter15 Phase 6 C1+C2 (2026-05-16): snapshot motion_aux head init + build
    # probe_clips for in-training trio. Same wiring as m09a2 (mirrors m09c1
    # encoder-side flow). probe is universal-true post yaml-parity audit.
    # variant_tag for in-training plot titles — DECLARED in head yaml data block
    # (single source; was a hardcoded .replace string-strip). KeyError = fail loud.
    _variant_tag = cfg["data"]["variant_tag"]
    head_init_params = {n: p.detach().cpu().clone() for n, p in ma_head.named_parameters()}
    head_drift_history = []
    probe_history = []
    # iter15 D15 (2026-05-16): probe-trio encoder cache. Encoder + predictor are
    # FROZEN for the entire run → encoder forward + future_l1 outputs are time-
    # invariant across val checkpoints. First call populates this dict; subsequent
    # calls reuse the cached pooled features + per-clip L1 → MA head re-runs each
    # time (it IS the trained component). Saves ~10 min × (N_val - 1) val cycles.
    # compute_metric_trio FAILS LOUD if encoder signature changes mid-run.
    encoder_cache: dict = {}
    probe_history_path = output_dir / "probe_history.jsonl"
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)} (for in-training trio future_l1)")
    probe_clips, probe_labels = None, None
    # probe block is declared in the yaml chain (single source) — read directly.
    _probe_block = cfg["probe"]
    if _probe_block["enabled"]:
        action_labels_path = (args.probe_action_labels or
                              str(Path(_probe_block["subset"]).parent / "action_labels.json"))
        if not Path(action_labels_path).exists():
            print(f"❌ FATAL [probe]: action_labels.json not found at {action_labels_path}", file=sys.stderr)
            sys.exit(3)
        probe_labels = load_action_labels(Path(action_labels_path))
        print(f"  [probe] decoding clips from {_probe_block['subset']} ...", flush=True)
        probe_clips = build_probe_clips(
            probe_subset_path=_probe_block["subset"],
            probe_local_data=_probe_block["local_data"],
            probe_tags_path=_probe_block["tags_path"],
            num_frames=cfg["data"]["num_frames"],
            crop_size=cfg["model"]["crop_size"],
            max_clips=cfg["monitoring"]["knn_probe_clips"],
        )
        print(f"  [probe] decoded {len(probe_clips)} clips ({len(probe_labels)} have action labels)", flush=True)

    # === Single head-only stage (validated in merge_config_with_args) ===
    surgery_cfg = cfg["surgery"]
    stage_cfg = surgery_cfg["stages"][0]
    stage_name = stage_cfg["name"]
    mode_mixture = stage_cfg["mode_mixture"]
    max_epochs_pct = stage_cfg["max_epochs_pct"]
    if abs(max_epochs_pct - 1.0) > 1e-6:
        print(f"FATAL [m09c2]: head-only mode requires max_epochs_pct=1.0 in the single stage; "
              f"got {max_epochs_pct}.")
        sys.exit(1)

    max_epochs = opt_cfg["max_epochs"]
    batch_size = opt_cfg["batch_size"]
    n_train = len(train_keys)
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = max_epochs * steps_per_epoch
    # iter15 D13 (2026-05-16): mirror m09a1:538 cadence — val/probe-trio every
    # val_interval steps (= steps_per_epoch // saves_per_epoch). Previously head
    # cells validated ONLY at end-of-epoch (1×/ep) while encoder cells used 2×/ep
    # → asymmetric trajectory plots. Aligns observability with m09a2 + m09a1/c1.
    saves_per_epoch = cfg["checkpoint"]["saves_per_epoch"]
    val_interval = max(1, steps_per_epoch // saves_per_epoch)
    print(f"Stage: {stage_name} · mixture: {mode_mixture} · mode: {mode_key} · "
          f"epochs: {max_epochs} · batch: {batch_size} · steps/epoch: {steps_per_epoch}")
    print(f"Validation every {val_interval} steps ({saves_per_epoch}x/epoch — D13 cadence parity)")

    scheduler = build_scheduler(optimizer, opt_cfg, total_steps)

    mp_cfg = cfg["mixed_precision"]
    # Supported mixed-precision dtypes (whitelist mirrors base_optimization.yaml
    # mixed_precision.dtype). KeyError = fail loud on a bad/typo'd dtype string —
    # was a ternary that SILENTLY defaulted to float16 on any non-bfloat16 value.
    # Unified across m09a1/a2/c1/c2.
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[mp_cfg["dtype"]]
    # iter15 Phase 5 V2 fix (2026-05-15): GradScaler required for motion_aux
    # backward (motion_aux_loss.py:413). No-op for bfloat16 default but the
    # object must exist. Mirrors m09a1:560-561.
    use_scaler = mp_cfg["enabled"] and mp_cfg["dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    sizer = AdaptiveBatchSizer(
        initial_size=batch_size,
        min_size=1,
        max_size=batch_size,
        memory_cap=_pcfg["gpu"]["gpu_memory_target"],
    )
    print(f"AdaptiveBatchSizer: {sizer}")

    mode_label = mode_key.upper()
    wb_run = init_wandb("m09c2", mode_label, config=vars(args),
                          enabled=not args.no_wandb)

    # === Build factor-aug DataLoader (replaces m09a2's producer_thread) ===
    loader = _build_factor_loader(
        cfg=cfg,
        train_keys=train_keys,
        mode_mixture=mode_mixture,
        stage_steps=total_steps,
        base_seed=seed,
    )

    # iter15 Phase 6 audit (2026-05-16): unified emit-set with m09c1 — renamed
    # training_log.jsonl → loss_log.jsonl + added loss_log.csv (m09c1 schema).
    # Frozen-encoder fields (loss_jepa/loss_drift/loss_multi_task) emit NaN/0.
    jsonl_path = output_dir / "loss_log.jsonl"
    summary_path = output_dir / "training_summary.json"
    train_log_f = jsonl_path.open("a", buffering=1)
    csv_path = output_dir / "loss_log.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["step", "epoch", "loss_jepa", "loss_drift", "loss_total",
                             "loss_multi_task", "loss_motion_aux",
                             "lr", "grad_norm", "throughput", "val_loss", "stage"])
        csv_file.flush()

    best_val_loss = float("inf")
    best_epoch = -1
    pbar = make_pbar(total=total_steps, desc=f"m09c2 head-only [{stage_name}]", unit="step")
    step = 0
    t_start = time.time()

    try:
        loader_iter = iter(loader)
        for epoch in range(max_epochs):
            ma_head.train()
            student.eval()
            epoch_train_losses = []
            epoch_started = time.time()

            for _ in range(steps_per_epoch):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    # DataLoader exhausted before stage budget — re-iterate.
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                batch_clips = batch["tensor"].to(device)            # (B, T, C, H, W)
                batch_clips = batch_clips.permute(0, 2, 1, 3, 4)    # (B, C, T, H, W)
                _ck = batch["clip_key"]
                batch_keys = list(_ck) if not isinstance(_ck, list) else _ck

                try:
                    optimizer.zero_grad(set_to_none=True)
                    loss_val, per_branch = run_motion_aux_step(
                        student, ma_head, ma_cfg, ma_lookup,
                        batch_clips, batch_keys, scaler=scaler,
                        mp_cfg=mp_cfg, dtype=dtype, device=device,
                    )
                    optimizer.step()
                    scheduler.step()
                    sizer.after_batch_success()
                except torch.cuda.OutOfMemoryError:
                    print(f"[m09c2] OOM at step {step}, sub-batch {sizer.size}")
                    cuda_cleanup()
                    if not sizer.on_oom():
                        print("FATAL [m09c2]: OOM at min sub-batch — cannot continue")
                        sys.exit(1)
                    continue

                epoch_train_losses.append(float(loss_val))
                step += 1
                pbar.update(1)
                if step % 20 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    row = {
                        "step": step, "epoch": epoch,
                        "loss_jepa": float("nan"),       # encoder frozen → no JEPA gradient
                        "loss_drift": 0.0,                # encoder frozen → drift ≡ 0
                        "loss_multi_task": float("nan"),  # multi_task off in head-only
                        "loss_motion_aux": float(loss_val),
                        "loss_total": float(loss_val),    # head-only total = motion_aux
                        "lr": cur_lr,
                        "branch": per_branch,
                        "stage": stage_name,
                    }
                    train_log_f.write(json.dumps(row) + "\n")
                    train_log_f.flush()
                    os.fsync(train_log_f.fileno())
                    csv_writer.writerow([step, epoch, "", "0", f"{float(loss_val):.6f}",
                                          "", f"{float(loss_val):.6f}",
                                          f"{cur_lr:.6e}", "", "", "", stage_name])
                    csv_file.flush()
                    log_metrics(wb_run, {"train/loss": float(loss_val),
                                         "train/lr": cur_lr},
                                step=step)

                # iter15 D13 (2026-05-16) + D16 fix (2026-05-16): val cycle every
                # val_interval steps. Mirrors m09a1/c1 cadence (saves_per_epoch=2 →
                # 2 probes per epoch). D16: removed the `step % steps_per_epoch == 0`
                # OR-clause that fired EXTRA val cycles when steps_per_epoch was not
                # evenly divisible by saves_per_epoch (e.g. POC m09c2: 35 steps/ep,
                # val_interval=17 → step 35 fired 1 step after step 34, wasting an
                # 11-min probe-trio cycle). The val_interval cadence alone gives
                # exactly saves_per_epoch fires per epoch (last fire near, but not
                # exactly at, end-of-epoch when there's a remainder).
                if step > 0 and step % val_interval == 0:
                    mean_train = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
                    val_loss = compute_val_motion_aux_loss(
                        student, ma_head, ma_cfg, ma_lookup,
                        val_keys, args.val_local_data, cfg, device, dtype,
                        tmp_prefix="m09c2_val_",
                    )
                    elapsed = time.time() - epoch_started
                    is_end_of_epoch = (step % steps_per_epoch == 0)
                    tag = "epoch-end" if is_end_of_epoch else "mid-epoch"
                    print(f"\n[step {step} · epoch {epoch}/{max_epochs} · {tag}] "
                          f"train_loss={mean_train:.4f}  val_loss={val_loss:.4f}  "
                          f"wall={elapsed:.0f}s  stage={stage_name}")
                    row = {"epoch": epoch, "train_loss": mean_train, "val_loss": val_loss,
                           "wall_sec": elapsed, "step": step, "stage": stage_name}
                    train_log_f.write(json.dumps(row) + "\n")
                    train_log_f.flush()
                    os.fsync(train_log_f.fileno())
                    csv_writer.writerow([step, epoch, "", "", "", "", "",
                                          "", "", "", f"{val_loss:.6f}", stage_name])
                    csv_file.flush()
                    log_metrics(wb_run, {"val/loss": val_loss,
                                         "train/epoch_mean_loss": mean_train,
                                         "epoch": epoch}, step=step)

                    val_total = val_loss * float(ma_cfg["weight_motion"])
                    probe_record = {
                        "step": step, "epoch": epoch,
                        "val_jepa_loss":       val_total,
                        "val_motion_aux_loss": val_loss,
                        "val_total_loss":      val_total,
                        "val_drift_loss":      0.0,
                        "val_multi_task_loss": 0.0,
                        "stage":               stage_name,
                        "epoch_pct": round((epoch + 1) / max_epochs * 100, 1),
                    }
                    if probe_clips is not None and probe_labels:
                        run_trio_at_val(
                            student, predictor, probe_clips, probe_labels,
                            mask_gen=mask_generators[0], cfg=cfg, device=device,
                            step=step, wb_run=wb_run, probe_record=probe_record,
                            motion_aux_head=ma_head,
                            encoder_cache=encoder_cache,   # iter15 D15
                            encoder_frozen=True,           # iter15 D15: head cell, encoder+predictor FROZEN
                        )
                    track_head_drift_at_val(
                        ma_head, head_init_params, head_drift_history,
                        output_dir=output_dir, step=step,
                        probe_record=probe_record,
                        title_prefix=f"m09c head [{_variant_tag}] step={step} · ",
                        file_prefix="m09c2",
                    )
                    probe_history.append(probe_record)
                    with open(probe_history_path, "a") as _ph:
                        _ph.write(json.dumps(probe_record) + "\n")
                        _ph.flush()
                        os.fsync(_ph.fileno())

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        export_motion_aux_head(ma_head, output_dir / "motion_aux_head.pt")
                        print(f"  ✅ new best val_loss={best_val_loss:.4f} (step {step}, epoch {epoch})")

                    # iter15 D17 (2026-05-16): symmetric per-val plot rendering with
                    # m09c1 encoder cell (which renders 5 plots per val: train_loss,
                    # loss_decomposition, block_drift, val_loss_jepa, probe_trajectory
                    # _trio). Head cells previously rendered ONLY block_drift per val
                    # (via track_head_drift_at_val) → 4 plots missing mid-training.
                    # Asymmetric observability between encoder + head sibling modules.
                    # End-of-training plots stay (idempotent, regenerates final view).
                    # FAIL LOUD on render exceptions per CLAUDE.md.
                    best_state_live = {
                        "step": step if best_epoch == epoch else -1,
                        "val_loss_at_best": float(best_val_loss),
                        "probe_top1": float(probe_record.get("probe_top1", -1.0)),
                    }
                    kill_state_live = {"triggered": False, "reason": None}
                    try:
                        plot_training_curves(
                            runs=[{"csv_path": str(csv_path),
                                   "label": f"m09c2 head-only [{_variant_tag}]",
                                   "color": "blue",
                                   "batch_size": batch_size}],
                            output_dir=str(output_dir),
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · {len(train_keys):,} train × "
                                         f"{max_epochs} ep × BS={batch_size} · step={step}\n",
                            file_prefix="m09c2",
                        )
                        plot_combined_losses(
                            jsonl_path=jsonl_path,
                            output_dir=output_dir,
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · LR={opt_cfg['lr']:.1e} · step={step} · ",
                            file_prefix="m09c2",
                        )
                        plot_val_loss_with_kill_switch_overlay(
                            probe_history, output_dir,
                            best_state=best_state_live, kill_state=kill_state_live,
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · val_total (motion_aux × weight) · step={step}\n",
                            file_prefix="m09c2",
                        )
                        plot_probe_trajectory_trio(
                            probe_history, output_dir,
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · step={step} · ",
                            file_prefix="m09c2",
                        )
                    except Exception as _e:
                        print(f"  [plot] FATAL: per-val plot render failed at step {step}: {_e}", flush=True)
                        print("  [plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                        raise
    finally:
        train_log_f.close()
        csv_file.close()

    pbar.close()

    # iter15 Phase 6 audit (2026-05-16): emit-parity plots — match m09c1.
    # _variant_tag already hoisted above (used by in-training trio plots too).
    try:
        plot_training_curves(
            runs=[{"csv_path": str(csv_path),
                   "label": f"m09c2 head-only [{_variant_tag}]",
                   "color": "blue",
                   "batch_size": batch_size}],
            output_dir=str(output_dir),
            title_prefix=f"m09c2 head-only [{_variant_tag}] · {len(train_keys):,} train × "
                         f"{max_epochs} ep × BS={batch_size}\n",
            file_prefix="m09c2",
        )
    except Exception as e:
        print(f"  [plot] WARN train_loss render skipped: {type(e).__name__}: {e}", flush=True)
    try:
        plot_combined_losses(
            jsonl_path=jsonl_path,
            output_dir=output_dir,
            title_prefix=f"m09c2 head-only [{_variant_tag}] · ",
            file_prefix="m09c2",
        )
    except Exception as e:
        print(f"  [plot] WARN loss_decomposition render skipped: {type(e).__name__}: {e}", flush=True)
    # iter15 Phase 6 C1+C3 (2026-05-16): probe_trajectory_trio + val_loss_jepa
    # plots from probe_history. track_head_drift_at_val (C2) already emits the
    # m09c_block_drift.{png,pdf} + m09c_block_drift_history.json per val cycle.
    # probe.enabled is universal-true (yaml audit 2026-05-16) → probe_history is
    # always non-empty by the time we reach this line.
    plot_probe_trajectory_trio(
        probe_history, output_dir,
        title_prefix=f"m09c2 head-only [{_variant_tag}] · ",
        file_prefix="m09c2",
    )
    kill_state = {"triggered": False, "reason": None}
    # D4 fix (2026-05-16): plot reads "val_loss_at_best" (plots.py:857), not
    # "val_jepa_loss_at_best". Use the schema the plot expects.
    best_state = {"step": -1, "val_loss_at_best": float("inf"), "probe_top1": -1.0}
    plot_val_loss_with_kill_switch_overlay(
        probe_history, output_dir,
        best_state=best_state, kill_state=kill_state,
        title_prefix=f"m09c2 head-only [{_variant_tag}] · val_total\n",
        file_prefix="m09c2",
    )

    # === Finalization ===
    student_export = output_dir / "student_encoder.pt"
    export_student_for_eval(student, student_export, explora_enabled=False)

    combined_ckpt = output_dir / f"{CHECKPOINT_PREFIX}_best.pt"
    torch.save({
        "student_state_dict": student.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "motion_aux_head_state_dict": ma_head.state_dict(),
        "n_motion_classes": ma_head.n_motion_classes,
        "n_motion_dims": ma_head.n_motion_dims,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stage_name": stage_name,
        "mode_mixture": mode_mixture,
        "type": "m09c2_head_only_surgery",
    }, combined_ckpt)
    print(f"Saved: {combined_ckpt}")

    summary = {
        "mode": mode_key,
        "adapted_encoder": cfg["data"]["adapted_encoder"],
        "stage_name": stage_name,
        "mode_mixture": mode_mixture,
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "total_steps": step,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "wall_sec": round(time.time() - t_start, 1),
        "head_params": head_params,
        "init_ckpt_path": init_ckpt_path,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {summary_path}")

    finish_wandb(wb_run)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 HEAD-ONLY surgery (m09c2 — iter15 Phase 2).")
    add_m09_common_args(parser, require_val_data=True)
    # iter15 (2026-05-17): mirror m09c1:1783-1787 — REQUIRED init for Δ5 paired-
    # test validity. Accepts hf:// URI OR local path; dispatcher in build_model.
    parser.add_argument("--init-from-ckpt", type=str, required=True,
                        help="iter15 REQUIRED: init source for frozen encoder + "
                             "predictor. Accepts hf://<owner>/<repo>/<filename> OR "
                             "local filesystem path. Schema MUST be the prior-run "
                             "ckpt (m09a_ckpt_best.pt) carrying 'student' + "
                             "'predictor' keys — NOT the Meta baseline.")
    add_wandb_args(parser)
    args = parser.parse_args()

    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_merged_config(args.model_config, args.train_config)
    cfg = merge_config_with_args(cfg, args)

    if args.cache_policy == "2":
        wipe_output_dir(cfg["checkpoint"]["output_dir"], args.cache_policy,
                          label="m09c2 head-only surgery output dir")

    train(cfg, args)


if __name__ == "__main__":
    main()
