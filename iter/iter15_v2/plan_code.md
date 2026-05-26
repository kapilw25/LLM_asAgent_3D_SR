# iter15-v2 — plan_code: remaining tasks (detailed CODE plan)

Scope = the 5 open tasks after the 2026-05-26 session (hardcoded-values + getattr/.get +
`--subset-mode legacy` retire + refactor #29 landed). #30 dropped (false premise: build_model
ckpt-LOAD diverges — a1/a2 Meta `target_encoder`, c1/c2 `init_from_ckpt` hf:// dispatcher +
`student` schema → no shared loader). #31-34 BLOCKED by #35.

```text
┌─────┬────────────────────────────────────────────────┬──────────┬──────────────┐
│  #  │ TASK                                           │ STATUS   │ BLOCKED BY    │
├─────┼────────────────────────────────────────────────┼──────────┼──────────────┤
│ 29  │ compute_val_motion_aux_loss → utils.training   │ ✅ DONE  │ —            │
│ 30  │ build_model ckpt-LOAD → shared                 │ ❌ DROP  │ false premise │
│ 35  │ GPU-SANITY checkpoint (validate session edits) │ ◻ OPEN  │ new instance  │
│ 31  │ build_student_predictor (construction) → util  │ ◻ OPEN  │ #35           │
│ 32  │ TrainLogWriter (jsonl+csv setup) → util         │ ◻ OPEN  │ #35           │
│ 33  │ render_val_plots (4 plot calls) → utils.plots   │ ◻ OPEN  │ #35           │
│ 34  │ finalize_outputs (export+ckpt+summary) → util   │ ◻ OPEN  │ #35           │
└─────┴────────────────────────────────────────────────┴──────────┴──────────────┘
```

---

## #35 · GPU-SANITY checkpoint (gate for #31-34)

Run on the new RTX Pro 4000 / 6000 instance. Exact commands + grep gate = runbook §0.5.
Coverage: dtype whitelist · getattr→args.X · .get→cfg[k] · contract-guard deletions ·
`--subset-mode` legacy removal · factor_streaming.sanity yaml · #29 shared val fn.
GATE: all 6 subcmds (m09a1/a2/c1/c2 × DI+noDI yaml variants) reach train loop + ≥1 val cycle;
`grep -iE "FATAL|Traceback|KeyError|AttributeError|invalid choice"` EMPTY. Then mark #35 done
→ unblock #31-34.

---

## #31 · build_student_predictor → utils.training

NEW (utils/training.py, after `augment_clip_consistent`):

```python
def build_student_predictor(model_cfg: dict, data_cfg: dict):
    """Construct (student ViT, predictor) UNLOADED + UNFROZEN. Shared by m09a1/a2/c1/c2.
    Caller owns ckpt-load (Meta vs init_from_ckpt) + freeze (set_trainable_prefix vs
    requires_grad=False) + EMA-teacher (a1/c1 only). Construction kwargs are identical."""
    from utils.vjepa2_imports import get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1
    arch = model_cfg["arch"]; crop = model_cfg["crop_size"]
    student = get_vit_by_arch(arch)(
        img_size=(crop, crop), patch_size=model_cfg["patch_size"],
        num_frames=data_cfg["num_frames"], tubelet_size=model_cfg["tubelet_size"],
        use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
        use_rope=model_cfg["use_rope"],
        use_activation_checkpointing=model_cfg["use_activation_checkpointing"])
    pred_ctor = get_vit_predictor_2_1() if model_cfg["predict_all"] else get_vit_predictor()
    predictor = pred_ctor(
        img_size=(crop, crop), patch_size=model_cfg["patch_size"],
        num_frames=data_cfg["num_frames"], tubelet_size=model_cfg["tubelet_size"],
        embed_dim=model_cfg["embed_dim"], predictor_embed_dim=model_cfg["pred_embed_dim"],
        depth=model_cfg["pred_depth"], num_heads=model_cfg["pred_num_heads"],
        use_mask_tokens=True, num_mask_tokens=model_cfg["num_mask_tokens"],
        zero_init_mask_tokens=True, use_rope=model_cfg["use_rope"], uniform_power=False,
        use_sdpa=True, use_silu=False, wide_silu=True,
        use_activation_checkpointing=model_cfg["use_activation_checkpointing"],
        return_all_tokens=model_cfg["predict_all"])
    return student, predictor
```

NOTE: hardcoded bool kwargs (`use_sdpa=True` …) are pre-existing; if hardcode-audit extends
here, source from `model_cfg` first (separate task — do NOT bundle).

```text
┌──────┬─────────────────────────────┬───────────────────────────────────────────────────┐
│ FILE │ build_model lines (replace) │ STAYS IN CALLER (divergent — do NOT extract)      │
├──────┼─────────────────────────────┼───────────────────────────────────────────────────┤
│ a1   │ 215-216,220-244 + 338-343   │ Meta target_encoder load · EMA teacher deepcopy@331│
│ a2   │ 135-136,139-150 + 202-221   │ Meta target_encoder load · freeze-all@194-196      │
│ c1   │ 300-301,305-330 + 444-449   │ init_from_ckpt dispatcher · EMA teacher@428        │
│ c2   │ 185-186,189-200 + 296-306   │ init_from_ckpt dispatcher · set_trainable_prefix(0)│
└──────┴─────────────────────────────┴───────────────────────────────────────────────────┘
```

Each caller: `student, predictor = build_student_predictor(model_cfg, data_cfg)` THEN its own
load/freeze/teacher. Add `build_student_predictor` to each `from utils.training import (…)`.
VERIFY: 3-check ×5 files; `--SANITY` ×4 → `Student loaded: … keys = NN%` + `assert_encoder_frozen` pass.

---

## #32 · TrainLogWriter → utils.training

csv header DIVERGES → `columns` is a param (NOT hardcoded in the util):

```text
┌────────┬──────────────────────────────────────────────────────────────────────────┐
│ MODULE │ csv header (current line)                                                 │
├────────┼──────────────────────────────────────────────────────────────────────────┤
│ a1/a2  │ [step,epoch,loss_jepa,loss_drift,loss_total,loss_multi_task,loss_motion_  │
│  /c2   │   aux,lr,grad_norm,throughput,val_loss(,stage←c2 only)]  (a1:798 a2:416 c2:602)│
│ c1     │ [step,stage,loss_jepa,loss_masked,loss_context,…]  (c1:868) — DISTINCT     │
└────────┴──────────────────────────────────────────────────────────────────────────┘
```

NEW (utils/training.py):

```python
class TrainLogWriter:
    """Crash-safe loss log: JSONL (fsync/write) + CSV mirror. Shared mechanics; schema
    via `columns`. Replaces dup jsonl_path/csv_path/_log_step setup in all 4 trainers."""
    def __init__(self, output_dir: Path, columns: list[str]):
        self.jsonl = (output_dir / "loss_log.jsonl").open("a", buffering=1)
        csv_path = output_dir / "loss_log.csv"; new = not csv_path.exists()
        self.csv_path = csv_path
        self._cf = open(csv_path, "a", newline=""); self._cw = csv.writer(self._cf)
        if new: self._cw.writerow(columns); self._cf.flush()
    def log_jsonl(self, record: dict):
        self.jsonl.write(json.dumps(record) + "\n"); self.jsonl.flush(); os.fsync(self.jsonl.fileno())
    def log_csv(self, row: list): self._cw.writerow(row); self._cf.flush()
    def close(self): self.jsonl.close(); self._cf.close()
```

Caller edits: replace the jsonl/csv setup block + local `_log_step` (a1:786) with
`logw = TrainLogWriter(output_dir, COLUMNS)`; swap `_log_step(r)`→`logw.log_jsonl(r)`,
`csv_writer.writerow(...)`→`logw.log_csv(...)`, `train_log_f.close()/csv_file.close()`→`logw.close()`.
Keep each module's `COLUMNS` list local (the divergent part). VERIFY: `--SANITY` → `loss_log.jsonl`
+ `loss_log.csv` exist with the SAME header bytes as pre-refactor (diff header row).

---

## #33 · render_val_plots → utils.plots

4 calls in fixed order (c2:766/776/782/788 mid-val; 806/819/832/841 end). title/file prefix differ.

NEW (utils/plots.py):

```python
def render_val_plots(*, csv_path, jsonl_path, probe_history, output_dir,
                     title_prefix: str, file_prefix: str, label: str, color: str,
                     batch_size: int, lr: float, best_state: dict, kill_state: dict):
    """The 4 per-val plots in one call (training_curves, combined_losses,
    val_loss_kill_overlay, probe_trajectory_trio). FAIL LOUD on render exc (CLAUDE.md)."""
    plot_training_curves(runs=[{"csv_path": str(csv_path), "label": label,
        "color": color, "batch_size": batch_size}], output_dir=str(output_dir),
        title_prefix=title_prefix, file_prefix=file_prefix)
    plot_combined_losses(jsonl_path=jsonl_path, output_dir=output_dir,
        title_prefix=f"{title_prefix}LR={lr:.1e} · ", file_prefix=file_prefix)
    plot_val_loss_with_kill_switch_overlay(probe_history, output_dir,
        best_state=best_state, kill_state=kill_state,
        title_prefix=title_prefix, file_prefix=file_prefix)
    plot_probe_trajectory_trio(probe_history, output_dir,
        title_prefix=title_prefix, file_prefix=file_prefix)
```

Caller edits: replace BOTH the mid-val `try: plot_*… except: raise` block AND the end-of-train
block with one `render_val_plots(...)` call each. Pass `file_prefix="m09a2"`/`"m09c2"`/etc.
a1/c1 also render `block_drift`/`val_loss_jepa` — keep those EXTRA calls in-caller (encoder-only).
VERIFY: `--SANITY` → same PNG/PDF set per `outputs/<mode>/<cell>/` as pre-refactor (`ls *.png`).

---

## #34 · finalize_outputs → utils.training

combined-ckpt payload DIVERGES (a2:no stage; c2:+stage_name+mode_mixture) → extra fields are params:

NEW (utils/training.py):

```python
def finalize_outputs(*, student, output_dir: Path, ckpt_prefix: str,
                     ckpt_payload: dict, summary: dict, explora_enabled: bool = False):
    """student_encoder.pt export + <prefix>_best.pt combined ckpt + training_summary.json.
    ckpt_payload / summary carry the per-module divergent fields (caller builds them)."""
    export_student_for_eval(student, output_dir / "student_encoder.pt",
                            explora_enabled=explora_enabled)
    combined = output_dir / f"{ckpt_prefix}_best.pt"
    torch.save(ckpt_payload, combined); print(f"Saved: {combined}")
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
```

Caller edits (a2:749-786, c2:906-942 region): build `ckpt_payload`/`summary` dict locally
(divergent keys stay there), then `finalize_outputs(student=…, ckpt_prefix=CHECKPOINT_PREFIX,
ckpt_payload=…, summary=…)`. Keep `os._exit(0)` + `finish_wandb` in caller.
VERIFY: `--SANITY` → `student_encoder.pt` + `m09{a,c}_ckpt_best.pt` + `training_summary.json`;
`torch.load(...).keys()` matches pre-refactor payload per module.

---

## Cross-cutting verify (after #31-34)

```bash
python -m py_compile src/utils/training.py src/utils/plots.py src/m09a1_pretrain_encoder.py \
  src/m09a2_pretrain_head.py src/m09c1_surgery_encoder.py src/m09c2_surgery_head.py
ruff check --select F,E9 src/utils/ src/m09a1_pretrain_encoder.py src/m09a2_pretrain_head.py \
  src/m09c1_surgery_encoder.py src/m09c2_surgery_head.py   # 0 unused imports
# then re-run #35 GPU-SANITY gate (runbook §0.5) — output file-sets MUST be byte-identical
```
