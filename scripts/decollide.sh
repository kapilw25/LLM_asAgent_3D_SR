#!/bin/bash
: '
=============================================================================
decollide.sh — end the macOS case-collision in factorjepa
=============================================================================

THE PROBLEM
    origin/main carries two different model arms whose paths differ only by
    one letter s case:

        vjepa_2_1_vitG   2B ViT-G  (champion / default_backbone)
        vjepa_2_1_vitg   1B ViT-g  (scale-axis ablation)

    348 tracked paths exist in both spellings with DIFFERENT content. macOS
    APFS is case-insensitive, so it can only hold one of each pair: during
    checkout the second write lands on the first file. The result is a working
    tree that no "git reset --hard" can ever make clean, and a "git add ."
    that would push one arm s bytes over the other s on GitHub.

THE FIX
    Rename the 1B arm to the suffixed form the codebase ALREADY uses elsewhere
    (src/utils/output_paths.py builds vjepa_2_1_vitg_1B / vjepa_2_1_vitG_2B,
    and src/utils/migrate_output_tree.py migrated outputs/ to exactly that):

        iter/**/vjepa_2_1_vitg/**            ->  iter/**/vjepa_2_1_vitg_1B/**
        **/scale_poc_vs_full_vjepa_2_1_vitg.{pdf,png}
                                             ->  ..._vjepa_2_1_vitg_1B.{pdf,png}

    The 2B arm keeps its name, so default_backbone and every config that names
    vjepa_2_1_vitG stay valid.

WHY THIS RUNS ON THE MAC, NOT THE GPU
    The rename is done entirely in git objects via a TEMPORARY INDEX. No file
    is ever written to disk, so the case-insensitive filesystem is never asked
    to hold both spellings. You do not need a GPU instance for this.

USAGE
    bash decollide.sh              # build + verify the commit, push NOTHING
    bash decollide.sh --push       # same, then push it to origin/main

    Afterwards, on the Mac:
        git fetch origin && git reset --hard origin/main
    and the working tree finally comes out clean.
=============================================================================
'

set -euo pipefail

REPO="/Users/kapilwanaskar/Downloads/research_projects/factorjepa"
BRANCH="main"
OLD_DIR="vjepa_2_1_vitg"
NEW_DIR="vjepa_2_1_vitg_1B"
DO_PUSH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --push) DO_PUSH=1 ;;
        *) echo "Unknown arg: $1"; echo "Usage: bash decollide.sh [--push]"; exit 1 ;;
    esac
    shift
done

cd "$REPO"

echo "[1/6] fetch origin/$BRANCH"
git fetch origin "$BRANCH" --quiet
BASE=$(git rev-parse "origin/$BRANCH")
echo "      base = $BASE"

echo "[2/6] count collisions in the base tree"
before=$(git ls-tree -r --name-only "$BASE" | awk '{print tolower($0)}' | sort | uniq -d | wc -l | tr -d ' ')
echo "      case-colliding paths: $before"
if [ "$before" -eq 0 ]; then
    echo "      nothing to do — origin/$BRANCH is already collision-free"
    exit 0
fi

# A temporary index means the real index and the working tree are never
# touched, so nothing is ever written to the case-insensitive filesystem.
TMP_INDEX=$(mktemp -t decollide-index)
trap 'rm -f "$TMP_INDEX"' EXIT
export GIT_INDEX_FILE="$TMP_INDEX"

echo "[3/6] load the base tree into a temporary index"
git read-tree "$BASE"

echo "[4/6] stage the renames"
n_dir=0
n_file=0
while IFS=$'\t' read -r meta path; do
    mode=$(printf '%s' "$meta" | awk '{print $1}')
    sha=$(printf '%s' "$meta" | awk '{print $2}')

    case "$path" in
        */$OLD_DIR/*)
            new="${path//\/$OLD_DIR\//\/$NEW_DIR\/}"
            n_dir=$((n_dir + 1)) ;;
        *_$OLD_DIR.pdf|*_$OLD_DIR.png)
            new="${path%_$OLD_DIR.*}_$NEW_DIR.${path##*.}"
            n_file=$((n_file + 1)) ;;
        *)
            continue ;;
    esac

    git update-index --add --cacheinfo "$mode,$sha,$new"
    git update-index --force-remove "$path"
    # git ls-files -s emits "<mode> <sha> <stage>\t<path>" — already tab-split.
done < <(git ls-files -s)

echo "      $n_dir path(s) under $OLD_DIR/  ->  $NEW_DIR/"
echo "      $n_file file(s) *_$OLD_DIR.{pdf,png}  ->  *_$NEW_DIR.{pdf,png}"

echo "[5/6] write the tree and verify it is collision-free"
TREE=$(git write-tree)
after=$(git ls-tree -r --name-only "$TREE" | awk '{print tolower($0)}' | sort | uniq -d | wc -l | tr -d ' ')
echo "      case-colliding paths after: $after"
if [ "$after" -ne 0 ]; then
    echo "      FATAL: renames did not remove every collision. Nothing pushed."
    git ls-tree -r --name-only "$TREE" | awk '{print tolower($0)"\t"$0}' | sort \
        | awk -F'\t' '{ if ($1==p1) print "        "$2; p1=$1 }' | head -10
    exit 1
fi

# Blob count must be identical: this is a pure rename, no content may be lost.
b_before=$(git ls-tree -r --name-only "$BASE" | wc -l | tr -d ' ')
b_after=$(git ls-tree -r --name-only "$TREE" | wc -l | tr -d ' ')
echo "      tracked files: $b_before -> $b_after"
if [ "$b_before" -ne "$b_after" ]; then
    echo "      FATAL: file count changed — a rename collided. Nothing pushed."
    exit 1
fi

MSG="refactor(paths): rename the 1B arm to vjepa_2_1_vitg_1B so it stops colliding with vitG

vjepa_2_1_vitG (2B ViT-G, default_backbone) and vjepa_2_1_vitg (1B ViT-g scale
axis) differ only by the case of one letter. 348 tracked paths existed in both
spellings with different content, which a case-insensitive filesystem cannot
represent: on macOS one arm overwrites the other during checkout, the working
tree can never be made clean, and git add . would push one arm bytes over the
other.

Rename the 1B arm to the suffixed form the codebase already uses elsewhere
(output_paths.bb_dir builds vjepa_2_1_vitg_1B / vjepa_2_1_vitG_2B, and
migrate_output_tree.py migrated outputs/ to exactly that). The 2B arm keeps its
name, so default_backbone and every config naming vjepa_2_1_vitG stay valid.

Pure rename: no blob changed, file count unchanged, collisions 348 -> 0."

COMMIT=$(git commit-tree "$TREE" -p "$BASE" -m "$MSG")
echo "[6/6] commit built: $COMMIT"

if [ "$DO_PUSH" -eq 0 ]; then
    echo
    echo "Nothing pushed (no --push). To inspect it:"
    echo "    git show --stat $COMMIT | head -30"
    echo "To publish:"
    echo "    bash $0 --push"
    echo
    echo "NOTE: the commit object lives in your local repo. If you do not push"
    echo "      it, git gc will eventually discard it — just re-run this script."
    exit 0
fi

unset GIT_INDEX_FILE
echo "      pushing $COMMIT -> origin/$BRANCH"
git push origin "$COMMIT:refs/heads/$BRANCH"

echo
echo "Pushed. Now bring this Mac in line:"
echo "    git fetch origin && git reset --hard origin/$BRANCH"
echo "The working tree should finally come out clean."
