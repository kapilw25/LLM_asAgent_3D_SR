┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚨 You're right — the prior 395 GB number was STEADY-STATE only (final ckpts + cleaned).   │
│    Training-time transients add a per-cell 35-56 GB peak that I did not call out.          │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│ Confirmed via configs/train/base_optimization.yaml:319  →  keep_last_n: 5                   │
│   ⇒ 5 periodic ckpts × ~7 GB each = 35 GB transient DURING training, alongside              │
│     best.pt (14 GB enc / 7 GB head) + last.pt (~7 GB) + final student_encoder.pt (6.9 GB)   │
└────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Per-cell footprint  ·  DURING training (peak)  vs  AFTER cleanup (settled)                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Cell type           │ DURING peak                                            │ AFTER settled       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 🔓 encoder-update   │ best 14 + last 7 + 5×periodic 35 + student 6.9 = 63 GB │ best 14 + student 7│
│                     │                                                        │              = 21 GB│
│ 🧊🧠 head-only      │ best 7 + last 6 + 5×periodic 30 + student 6.9 = 50 GB  │ best 7 + student 7 │
│                     │                                                        │              = 14 GB│
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Scenario A · ONE instance, ONE cell at a time (sequential, as runbook is written)                            │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Source data shared (A)                                                              204 GB persistent        │
│ Peak transient during heaviest cell (1 enc cell mid-train)                          +63 GB transient         │
│ Ckpts already-settled from prior cells (worst case: 4 enc + 2 head before 7th)     +4×21 + 2×14 = 112 GB    │
│ Eval probes (after run_eval)                                                         +65 GB                  │
│ ───────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│ PEAK moment in run                                                                  ~444 GB                  │
│ FINAL steady-state after all 7 cells complete + run_eval cleanup                    ~395 GB                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Scenario B · ONE instance, 3 cells running in PARALLEL (single machine, 3 processes)                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Source data shared once                                                              204 GB persistent       │
│ 3 enc cells in-flight: 3 × 63 GB                                                    +189 GB transient        │
│   (or 3 head cells: 3 × 50 GB = +150 GB)                                                                     │
│ Already-settled ckpts from any earlier cells                                         +0-112 GB               │
│ Eval probes (only after all cells done, so 0 at this peak)                          +0 GB                    │
│ ───────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│ PEAK moment (3 enc parallel, no prior settled)                                      ~393 GB                  │
│ PEAK moment (3 enc parallel, mid-pipeline w/ 3 cells settled)                       ~456 GB                  │
│ ⚠️  Plus parallel also multiplies the I/O + VRAM cost — 3 cells × 96 GB VRAM each =                          │
│    needs 3 GPUs (the 96 GB Blackwell can't host 3 enc trainings on one card).                                │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Scenario C · 3 INDEPENDENT instances on the same physical disk (3 machines / 3 users / 3 jobs)               │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Source data CAN be shared via symlink → outputs/full_local/ mounted read-only        204 GB shared           │
│ But each instance writes its own outputs/full/ tree (DIFFERENT ckpts/probes)                                  │
│   3 × 126 GB ckpts                                                                  +378 GB                  │
│   3 × 65 GB eval probes                                                              +195 GB                  │
│   3 × 63 GB transient peak (one cell mid-train per instance)                         +189 GB                  │
│ ───────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│ PEAK total                                                                          ~966 GB                  │
│ FINAL steady-state (each instance settled, no overlap)                              ~777 GB                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚨 FIT vs CURRENT  /dev/nvme1n1p2 = 299 GB (73 GB free, 227 GB used)                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Scenario A  ·  1 instance sequential        peak 444 / steady 395 GB    ❌ 1.3× over          │
│ Scenario B  ·  1 instance 3-parallel        peak 393-456 GB             ❌ 1.5× over          │
│ Scenario C  ·  3 instances on same disk     peak 966 GB                 ❌ 3.2× over          │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ All three scenarios overflow this disk.   Minimum disk needed:                              │
│   A: 500 GB nvme (single instance · FULL)                                                    │
│   B: 600 GB nvme (single instance · 3-cell parallel training)                                │
│   C: 1.0 TB nvme (3 instances, source-shared)  ·  3 × 256 GB if isolated mounts             │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
