# Safer two-step (recommended — preview before delete):

# 1. dry-run preview (sorted by size, human-readable)
find . -type f -size +50M -printf "%s\t%p\n" | sort -rn | numfmt --field=1 --to=iec

# 2. once you're happy with the list:
find . -type f -size +50M -delete

# 3. for manually delete
rm -rf <file1_path1> <file2_path1>

# ── Archive outputs/poc/ → iter15_v2 result_outputs, SKIPPING heavy binaries (ckpts + arrays) ──
# keeps .json/.png/.pdf/.csv/.jsonl/.log; drops .pt/.npy/.npz. -a preserves the tree.
rsync -av --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' --exclude='tmp_*' \
outputs/poc/ iter/iter17_ablations_model/result_outputs/v17b_train_eval/poc/


