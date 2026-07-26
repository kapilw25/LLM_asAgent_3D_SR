#!/bin/bash
: '
=============================================================================
Mac <-> GPU Sync — Claude sessions + repo, .env, git_push.sh
=============================================================================

PART 1 — THE SESSION PATH PROBLEM, AND HOW THIS SCRIPT RESOLVES IT
------------------------------------------------------------------
Claude Code files every session under a folder named after the project s
ABSOLUTE PATH, with every non-alphanumeric character replaced by "-":

    Mac   /Users/kapilwanaskar/factorjepa  ->  ~/.claude/projects/-Users-kapilwanaskar-factorjepa
    GPU   /workspace/factorjepa            ->  ~/.claude/projects/-workspace-factorjepa

Those two slugs can never be made to match:
  - macOS has a read-only root volume, so the repo cannot live at /workspace.
  - A symlink does not help. Claude Code is a Node app, and getcwd() resolves
    symlinks to the physical path, so it would still see /Users/... on the Mac.

So this script does not try to make the paths equal. It TRANSLATES them on
every transfer: the folder name is swapped, and every absolute path embedded
inside the transcripts is rewritten too. Each machine ends up holding a
native-looking copy of the same session, and "claude --resume" works on both.

    Mac  "cwd":"/Users/kapilwanaskar/factorjepa"  <-->  GPU  "cwd":"/workspace/factorjepa"

SAFETY MODEL — FAST-FORWARD ONLY, COMPARED IN CANONICAL SPACE
-------------------------------------------------------------
Session JSONLs are append-only, so a transfer is safe exactly when the
destination is a byte-for-byte PREFIX of the incoming file — the same test git
uses for a fast-forward push.

That comparison must NOT be done on raw bytes. A transcript may literally
MENTION the other machine s path (a conversation about this very script does).
Rewriting changes those mentions too, so an otherwise untouched file comes back
looking modified — a false "diverged" that would block syncing forever.

Every comparison therefore runs in CANONICAL space: both repo paths collapse to
one neutral token before comparing, which makes the test invariant to whichever
direction the file was last rewritten.

  - destination missing            -> copy
  - canonically identical          -> skip
  - destination is a canon prefix  -> fast-forward (old copy saved to backups)
  - anything else                  -> DIVERGED: destination kept, loud warning

Destination-only files are never deleted.

PART 2 — THE REPO
-----------------
"git fetch && git reset --hard origin/main" does NOT give a complete tree, for
three reasons specific to this repo:

  (a) Most of the bulk is not in git at all. outputs/, data/, deps/ and venv*
      are gitignored by design and live on HuggingFace — see
      src/utils/hf_outputs.py. data/eval_10k_local/tags.json (148 MB) was
      outright rejected by GitHub s 100 MB single-file limit. reset --hard can
      never materialise what was never pushed.
  (b) reset --hard does not delete untracked files. Afterwards the tree is
      "origin/main PLUS local leftovers", which is not "equal to origin/main".
      It also DISCARDS uncommitted edits to tracked files without asking.
  (c) The pack is 4.35 GiB over HTTPS. A fetch that stalls or takes an HTTP/2
      stream reset can leave origin/main stale, and "&&" does not always catch
      it — you then reset onto a commit that is not the real tip.

This script replaces that command with: retried fetch, a remote-SHA assertion
proving the fetch actually landed, reset, "git clean -fd" (no -x, so venv/,
data/, outputs/, deps/ and .env survive), then a post-condition check — and it
REFUSES to run at all while the working tree still has uncommitted changes.

USAGE
-----
    bash claude_session.sh --download [--host <alias>] [--no-repo|--force-repo]
        # GPU -> Mac. Run BEFORE destroying the GPU instance.
        # 1) pulls sessions   2) syncs the repo to origin/main

    bash claude_session.sh --upload [--host <alias>]
        # Mac -> GPU. Run on a fresh GPU instance.
        # Also pushes .env and git_push.sh (one-way, Mac -> GPU).
        # Refuses if the GPU is ahead — run --download first.

    bash claude_session.sh --verify-repo [--deep]
        # No SSH, no writes. Reports every discrepancy between the working
        # tree and origin/main, plus what is absent BY DESIGN (HF-hosted).
        # --deep also runs git fsck --connectivity-only.

    bash claude_session.sh --import-legacy
        # One-shot, no SSH. Ingests the old backup folder into the live Mac
        # sessions dir. Only needed once, to recover pre-rewrite backups.

    --host <alias>   SSH alias from ~/.ssh/config   (default: below)
    --dry-run        Show every decision, change nothing.
    --force          Overwrite sessions on divergence (old copy backed up).
    --no-repo        --download: skip the repo sync entirely.
    --force-repo     --download: reset the repo even with local changes.

Required SSH alias (Mac ~/.ssh/config):
    Host vast_RTXpro6000_96GB
        HostName <ip>
        User root
        Port <port>
        IdentityFile ~/.ssh/id_ed25519

Notes:
    - Runs ON THE MAC. Bypasses GitHub for sessions, so transcripts and .env
      never leave your two machines.
    - Overwritten files are kept under ~/.claude_session_backups/
    - A raw mirror of the GPU tree lives in ~/.claude_session_backups/mirror/
      so rsync only ships appended bytes.
    - SSH alias IP/port changes per instance — update ~/.ssh/config each time.

=============================================================================
'

set -euo pipefail

# === Configurable ===
SSH_HOST="vast_RTXpro6000_96GB"        # override per-run with --host <alias>
MAC_REPO="/Users/kapilwanaskar/Downloads/research_projects/factorjepa"
GPU_REPO="/workspace/factorjepa"
GIT_BRANCH="main"

# Gitignored ON PURPOSE and sourced from HuggingFace, not git. --verify-repo
# reports these separately so their absence never reads as a failed download.
HF_BACKED=("outputs" "data" "deps" "venv_walkindia")

# === Args ===
MODE=""
DRY_RUN=0
FORCE=0
DO_REPO=1
FORCE_REPO=0
DEEP=0
OLD_REPO=""

usage() {
    echo "Usage: bash claude_session.sh --download|--upload|--verify-repo|--import-legacy|--relocate <old_path> [options]"
    echo "  options: --host <alias>  --dry-run  --force  --no-repo  --force-repo  --deep"
    echo "  e.g. bash claude_session.sh --download --host vast_1x_12gb"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --upload|--download|--import-legacy|--verify-repo) MODE="$1" ;;
        --relocate)
            shift
            [ $# -gt 0 ] || { echo "Error: --relocate requires the OLD absolute repo path"; usage; exit 1; }
            MODE="--relocate"; OLD_REPO="$1" ;;
        --dry-run)    DRY_RUN=1 ;;
        --force)      FORCE=1 ;;
        --no-repo)    DO_REPO=0 ;;
        --force-repo) FORCE_REPO=1 ;;
        --deep)       DEEP=1 ;;
        --host)
            shift
            [ $# -gt 0 ] || { echo "Error: --host requires an SSH alias (from ~/.ssh/config)"; usage; exit 1; }
            SSH_HOST="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
    shift
done

[ -n "$MODE" ] || { echo "Error: a mode is required"; usage; exit 1; }

# === Derived ===
# Claude Code's slug rule: every non-alphanumeric character becomes "-".
slug() { printf '%s' "$1" | sed 's#[^A-Za-z0-9]#-#g'; }

MAC_SLUG="$(slug "$MAC_REPO")"
GPU_SLUG="$(slug "$GPU_REPO")"

MAC_SESSIONS="$HOME/.claude/projects/$MAC_SLUG"   # the LIVE dir claude --resume reads
GPU_SESSIONS_REL=".claude/projects/$GPU_SLUG"     # relative to the GPU's home dir

BACKUP_ROOT="$HOME/.claude_session_backups"
MIRROR="$BACKUP_ROOT/mirror/$GPU_SLUG"            # raw GPU tree, kept for rsync deltas
STAGE="$BACKUP_ROOT/.stage"                       # transient, path-rewritten tree

# Where the previous version of this script parked its backups. Its MAC_BASE
# pointed at a stale clone under Downloads, not at the live repo — which is
# also why it was shipping a two-month-old .env to the GPU.
LEGACY_BASE="/Users/kapilwanaskar/Downloads/research_projects/factorjepa_stale_jun24"
LEGACY_DIR="$LEGACY_BASE/.claude_sessions/projects/$GPU_SLUG"
MAC_ENV="$MAC_REPO/.env"
MAC_GITPUSH="$MAC_REPO/git_push.sh"

if [ "$(uname)" != "Darwin" ]; then
    echo "Warning: this script is meant to run on the Mac, not on $(uname). Continuing anyway."
fi

# =============================================================================
# Session helpers
# =============================================================================

fsize() { wc -c < "$1" | tr -d ' '; }

# canon <file> — emit the file with BOTH machines' repo paths collapsed to one
# neutral token. All merge comparisons happen in this space, so a path rewrite
# (or a transcript that literally mentions the other machine's path) can never
# masquerade as divergence.
canon() {
    CN_A="$MAC_REPO" CN_B="$GPU_REPO" LC_ALL=C perl -pe '
        s!\Q$ENV{CN_A}\E!\x01REPO\x01!g;
        s!\Q$ENV{CN_B}\E!\x01REPO\x01!g;
    ' < "$1"
}

canon_size() { canon "$1" | wc -c | tr -d ' '; }

canon_equal() { cmp -s <(canon "$1") <(canon "$2"); }

# canon_prefix <src> <dst> — true if canon(dst) is a prefix of canon(src).
# head -c closes the pipe early so perl takes SIGPIPE; pipefail is disabled
# inside the subshell to stop that from reading as a comparison failure.
canon_prefix() {
    local cs
    cs=$(canon_size "$2")
    if [ "$cs" -eq 0 ]; then return 0; fi
    ( set +o pipefail
      canon "$1" | head -c "$cs" | cmp -s - <(canon "$2") )
}

# rewrite_tree <src_dir> <stage_dir> <from_path> <to_path>
rewrite_tree() {
    local src="$1" stage="$2"
    export RW_FROM="$3" RW_TO="$4"

    rm -rf "$stage"; mkdir -p "$stage"
    ( cd "$src" && find . -type d ) | while IFS= read -r rd; do mkdir -p "$stage/$rd"; done
    ( cd "$src" && find . -type f ) | while IFS= read -r rf; do
        case "$rf" in
            *.jsonl|*.json|*.md|*.txt|*.js|*.log)
                LC_ALL=C perl -pe 's!\Q$ENV{RW_FROM}\E!$ENV{RW_TO}!g' < "$src/$rf" > "$stage/$rf" ;;
            *)
                cp -p "$src/$rf" "$stage/$rf" ;;
        esac
    done
}

# backup_file <dst_root> <relpath> — stash the version about to be overwritten
backup_file() {
    local root="$1" rel="$2"
    local bdir="$BACKUP_ROOT/$(basename "$root")/$(dirname "$rel")"
    mkdir -p "$bdir"
    cp -p "$root/$rel" "$bdir/$(basename "$rel")"
}

# merge_tree <src_dir> <dst_dir> — fast-forward-only. Sets DIVERGED.
DIVERGED=0
merge_tree() {
    local src="$1" dst="$2"
    local n_new=0 n_same=0 n_ff=0 n_div=0
    DIVERGED=0

    if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$dst"; fi
    ( cd "$src" && find . -type d ) | while IFS= read -r rd; do
        if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$dst/$rd"; fi
    done

    while IFS= read -r rel; do
        rel="${rel#./}"
        local s="$src/$rel" d="$dst/$rel" ss ds
        ss=$(fsize "$s")

        if [ ! -e "$d" ]; then
            printf '    new            %s  (%s B)\n' "$rel" "$ss"
            if [ "$DRY_RUN" -eq 0 ]; then cp -p "$s" "$d"; fi
            n_new=$((n_new + 1)); continue
        fi

        ds=$(fsize "$d")

        # Fast path: byte-identical needs no canonicalisation at all.
        if [ "$ss" -eq "$ds" ] && cmp -s "$s" "$d"; then
            n_same=$((n_same + 1)); continue
        fi

        if canon_equal "$s" "$d"; then
            n_same=$((n_same + 1)); continue
        fi

        if canon_prefix "$s" "$d"; then
            printf '    fast-forward   %s  (+%s B)\n' "$rel" "$((ss - ds))"
            if [ "$DRY_RUN" -eq 0 ]; then backup_file "$dst" "$rel"; cp -p "$s" "$d"; fi
            n_ff=$((n_ff + 1)); continue
        fi

        if [ "$FORCE" -eq 1 ]; then
            printf '    FORCED         %s  (src %s B, dst %s B) — dst overwritten, old copy backed up\n' "$rel" "$ss" "$ds"
            if [ "$DRY_RUN" -eq 0 ]; then backup_file "$dst" "$rel"; cp -p "$s" "$d"; fi
            n_ff=$((n_ff + 1)); continue
        fi

        printf '    DIVERGED       %s  (src %s B, dst %s B) — destination KEPT, nothing written\n' "$rel" "$ss" "$ds"
        n_div=$((n_div + 1))
    done < <( cd "$src" && find . -type f )

    echo "    ---- $n_new new, $n_ff fast-forwarded, $n_same unchanged, $n_div diverged"
    DIVERGED=$n_div
}

fetch_sessions() {
    ssh "$SSH_HOST" "test -d \$HOME/$GPU_SESSIONS_REL" \
        || { echo "FATAL: GPU has no sessions at ~/$GPU_SESSIONS_REL"; exit 1; }

    mkdir -p "$MIRROR"
    if ssh "$SSH_HOST" "command -v rsync >/dev/null 2>&1"; then
        rsync -az --partial -e ssh "$SSH_HOST:$GPU_SESSIONS_REL/" "$MIRROR/"
    else
        echo "    (rsync not on GPU — falling back to scp, full transfer)"
        scp -q -C -r "$SSH_HOST:$GPU_SESSIONS_REL/." "$MIRROR/"
    fi
}

# =============================================================================
# Repo helpers
#
# Every "git ... | head -N" is written as "| sed -n '1,Np'" on purpose: head
# closes the pipe early, git dies of SIGPIPE, and under pipefail+errexit that
# aborts the script. sed reads its input to completion, so it cannot.
# =============================================================================

REPO_DIRTY=0
REPO_BEHIND=0

# count_lines <captured-output> — "grep -c ." cannot be used here: on empty
# input it prints 0 AND exits 1, so a "|| printf 0" fallback fires as well and
# yields the two-line string "0\n0", which then crashes every [ -gt ] test.
count_lines() {
    if [ -z "$1" ]; then printf '0'; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi
}

verify_repo() {
    cd "$MAC_REPO"
    REPO_DIRTY=0
    REPO_BEHIND=0

    echo "  --- remote reachability ---"
    local remote_sha local_sha
    remote_sha=$(git ls-remote origin "refs/heads/$GIT_BRANCH" | cut -f1)
    if [ -z "$remote_sha" ]; then
        echo "    FATAL: origin/$GIT_BRANCH not readable — check network or credentials"
        exit 5
    fi
    if git rev-parse --verify --quiet "origin/$GIT_BRANCH" >/dev/null; then
        local_sha=$(git rev-parse "origin/$GIT_BRANCH")
    else
        local_sha="(never fetched)"
    fi
    echo "    github  origin/$GIT_BRANCH : $remote_sha"
    echo "    local   origin/$GIT_BRANCH : $local_sha"
    if [ "$remote_sha" != "$local_sha" ]; then
        echo "    STALE: the local remote-tracking ref is behind GitHub. This is"
        echo "           cause (c) — a fetch that silently did not land."
        REPO_BEHIND=1
    fi

    if [ "$local_sha" = "(never fetched)" ]; then
        echo "    -> nothing else can be compared until: git fetch origin"
        return 0
    fi

    echo "  --- commits ---"
    local counts ahead behind
    counts=$(git rev-list --left-right --count "HEAD...origin/$GIT_BRANCH")
    ahead=$(printf '%s' "$counts" | awk '{print $1}')
    behind=$(printf '%s' "$counts" | awk '{print $2}')
    echo "    local-only commits: $ahead   origin-only commits: $behind"
    if [ "$behind" -gt 0 ]; then REPO_BEHIND=1; fi

    echo "  --- tracked files differing from origin/$GIT_BRANCH ---"
    local difflist ndiff
    difflist=$(git diff --name-status "origin/$GIT_BRANCH")
    ndiff=$(count_lines "$difflist")
    echo "    $ndiff file(s)"
    if [ "$ndiff" -gt 0 ]; then
        REPO_DIRTY=1
        printf '%s\n' "$difflist" | sed -n '1,15p' | sed 's/^/      /'
        if [ "$ndiff" -gt 15 ]; then echo "      ... and $((ndiff - 15)) more"; fi
    fi

    echo "  --- files origin/$GIT_BRANCH has but disk does not ---"
    local misslist nmiss
    misslist=$(git diff --diff-filter=D --name-only "origin/$GIT_BRANCH")
    nmiss=$(count_lines "$misslist")
    echo "    $nmiss file(s)"
    if [ "$nmiss" -gt 0 ]; then printf '%s\n' "$misslist" | sed -n '1,10p' | sed 's/^/      /'; fi

    echo "  --- untracked and NOT ignored (git clean -fd would delete these) ---"
    local untlist nunt
    untlist=$(git ls-files --others --exclude-standard)
    nunt=$(count_lines "$untlist")
    echo "    $nunt file(s)"
    if [ "$nunt" -gt 0 ]; then printf '%s\n' "$untlist" | sed -n '1,10p' | sed 's/^/      /'; fi

    echo "  --- absent BY DESIGN (gitignored; restore from HuggingFace, not git) ---"
    local p n
    for p in "${HF_BACKED[@]}"; do
        n=$(count_lines "$(git ls-files "$p")")
        if [ -e "$p" ]; then
            printf '    %-16s on disk %-8s  tracked-in-git %s\n' "$p" "$(du -sh "$p" | cut -f1)" "$n"
        else
            printf '    %-16s ABSENT           tracked-in-git %s\n' "$p" "$n"
        fi
    done
    echo "    -> git will never deliver these. Use: python -u src/utils/hf_outputs.py download-data"

    if [ "$DEEP" -eq 1 ]; then
        echo "  --- deep object check (git fsck --connectivity-only) ---"
        if git fsck --connectivity-only --no-dangling; then
            echo "    OK: every object reachable from origin/$GIT_BRANCH is present"
        else
            echo "    FATAL: object graph is incomplete or corrupt — reclone required"
            exit 5
        fi
    fi

    echo "  --- verdict ---"
    if [ "$REPO_DIRTY" -eq 0 ] && [ "$REPO_BEHIND" -eq 0 ]; then
        echo "    IN SYNC: every tracked file matches origin/$GIT_BRANCH."
    else
        echo "    OUT OF SYNC: see above."
    fi
}

# sync_repo — reliable replacement for "git fetch && git reset --hard origin/main"
sync_repo() {
    cd "$MAC_REPO"

    echo "  [a] fetch (retried — the pack is multi-GB over HTTPS)"
    local ok=0 attempt
    for attempt in 1 2 3; do
        if [ "$DRY_RUN" -eq 1 ]; then ok=1; break; fi
        if git fetch origin --prune --tags --force; then ok=1; break; fi
        echo "      attempt $attempt failed — retrying in 3s"
        sleep 3
    done
    if [ "$ok" -eq 0 ]; then
        echo "      FATAL: fetch failed 3 times."
        echo "             Try: git config http.version HTTP/1.1   (HTTP/2 stream resets)"
        return 1
    fi

    echo "  [b] assert the fetch actually landed"
    local remote_sha local_sha
    remote_sha=$(git ls-remote origin "refs/heads/$GIT_BRANCH" | cut -f1)
    local_sha=$(git rev-parse "origin/$GIT_BRANCH")
    if [ "$remote_sha" != "$local_sha" ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "      FATAL: origin/$GIT_BRANCH is $local_sha but GitHub says $remote_sha"
        return 1
    fi
    echo "      origin/$GIT_BRANCH = $local_sha"

    echo "  [c] reset working tree to origin/$GIT_BRANCH"
    if [ "$DRY_RUN" -eq 0 ]; then git reset --hard "origin/$GIT_BRANCH"; fi

    # -fd, deliberately NOT -fdx. Without -x, gitignored paths survive, so
    # .env, venv_walkindia/, data/, outputs/ and deps/ are left untouched.
    echo "  [d] remove untracked, non-ignored leftovers (git clean -fd)"
    if [ "$DRY_RUN" -eq 1 ]; then
        git clean -nd | sed 's/^/      would /'
    else
        git clean -fd | sed 's/^/      /'
    fi

    echo "  [e] post-condition"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "      (dry run — not checked)"
    elif [ -z "$(git status --porcelain)" ] && git diff --quiet "origin/$GIT_BRANCH"; then
        echo "      VERIFIED: working tree is byte-identical to origin/$GIT_BRANCH"
    else
        echo "      FATAL: tree still differs from origin/$GIT_BRANCH — run --verify-repo"
        return 1
    fi
}

# =============================================================================
# Modes
# =============================================================================
case "$MODE" in

    --download)
        echo "=== Download: GPU ($SSH_HOST) -> Mac ==="
        echo "    $GPU_REPO  ->  $MAC_REPO"
        echo

        # Sessions first: they are irreplaceable and the instance may be about
        # to be destroyed. A repo problem must never cost you the transcripts.
        echo "[1/4] fetch  ~/$GPU_SESSIONS_REL  ->  mirror"
        fetch_sessions

        echo "[2/4] rewrite  $GPU_REPO  ->  $MAC_REPO"
        rewrite_tree "$MIRROR" "$STAGE" "$GPU_REPO" "$MAC_REPO"

        echo "[3/4] merge into $MAC_SESSIONS"
        merge_tree "$STAGE" "$MAC_SESSIONS"
        rm -rf "$STAGE"
        SESSION_DIVERGED=$DIVERGED

        REPO_RC=0
        if [ "$DO_REPO" -eq 0 ]; then
            echo "[4/4] repo sync SKIPPED (--no-repo)"
        else
            echo "[4/4] repo sync -> origin/$GIT_BRANCH"
            verify_repo
            echo
            if [ "$REPO_DIRTY" -eq 1 ] && [ "$FORCE_REPO" -eq 0 ]; then
                echo "    SKIPPED: the working tree has local changes (listed above)."
                echo "             git reset --hard would discard them."
                echo "             Commit them, or re-run with --force-repo."
                REPO_RC=1
            else
                sync_repo || REPO_RC=1
            fi
        fi

        echo
        if [ "$SESSION_DIVERGED" -gt 0 ]; then
            echo "WARNING: $SESSION_DIVERGED session file(s) diverged — both machines appended since the last sync."
            echo "         Nothing was overwritten. Inspect, then re-run with --force to take the GPU copy."
            exit 3
        fi
        echo "Sessions OK. On the Mac:  cd $MAC_REPO && claude --resume"
        if [ "$REPO_RC" -ne 0 ]; then
            echo "Repo NOT synced — see [4/4] above."
            exit 4
        fi
        ;;

    --upload)
        echo "=== Upload: Mac -> GPU ($SSH_HOST) ==="
        echo "    $MAC_REPO  ->  $GPU_REPO"
        echo

        [ -f "$MAC_ENV" ]      || { echo "FATAL: $MAC_ENV missing on Mac"; exit 1; }
        [ -f "$MAC_GITPUSH" ]  || { echo "FATAL: $MAC_GITPUSH missing on Mac"; exit 1; }
        [ -d "$MAC_SESSIONS" ] || { echo "FATAL: no Mac sessions at $MAC_SESSIONS"; exit 1; }

        ssh "$SSH_HOST" "test -d $GPU_REPO" \
            || { echo "FATAL: $GPU_REPO not present on GPU — clone factorjepa there first"; exit 1; }

        echo "[1/4] .env -> $GPU_REPO/.env"
        if [ "$DRY_RUN" -eq 0 ]; then scp -q "$MAC_ENV" "$SSH_HOST:$GPU_REPO/.env"; fi

        echo "[2/4] git_push.sh -> $GPU_REPO/git_push.sh"
        if [ "$DRY_RUN" -eq 0 ]; then
            scp -q "$MAC_GITPUSH" "$SSH_HOST:$GPU_REPO/git_push.sh"
            ssh "$SSH_HOST" "chmod +x $GPU_REPO/git_push.sh"
        fi

        # Pull the GPU's current state first so the fast-forward check runs
        # locally against real bytes — the GPU never runs any merge logic.
        echo "[3/4] check GPU state (mirror refresh)"
        mkdir -p "$MIRROR"
        if ssh "$SSH_HOST" "test -d \$HOME/$GPU_SESSIONS_REL"; then
            fetch_sessions
            rewrite_tree "$MAC_SESSIONS" "$STAGE" "$MAC_REPO" "$GPU_REPO"
            merge_tree "$STAGE" "$MIRROR"
            if [ "$DIVERGED" -gt 0 ]; then
                rm -rf "$STAGE"
                echo
                echo "ABORTED: $DIVERGED file(s) on the GPU are ahead of, or diverged from, the Mac."
                echo "         Run:  bash claude_session.sh --download --host $SSH_HOST"
                echo "         first, then upload. (--force overrides.)"
                exit 3
            fi
        else
            echo "    GPU has no sessions yet — first upload to this instance."
            rewrite_tree "$MAC_SESSIONS" "$STAGE" "$MAC_REPO" "$GPU_REPO"
            if [ "$DRY_RUN" -eq 0 ]; then
                rm -rf "$MIRROR"; mkdir -p "$MIRROR"; cp -R "$STAGE/." "$MIRROR/"
            fi
        fi
        rm -rf "$STAGE"

        echo "[4/4] push mirror -> ~/$GPU_SESSIONS_REL"
        if [ "$DRY_RUN" -eq 0 ]; then
            ssh "$SSH_HOST" "mkdir -p \$HOME/$GPU_SESSIONS_REL"
            if ssh "$SSH_HOST" "command -v rsync >/dev/null 2>&1"; then
                rsync -az --partial -e ssh "$MIRROR/" "$SSH_HOST:$GPU_SESSIONS_REL/"
            else
                scp -q -C -r "$MIRROR/." "$SSH_HOST:$GPU_SESSIONS_REL/"
            fi
        fi

        echo
        echo "Done. On the GPU:  cd $GPU_REPO && claude --resume"
        ;;

    --verify-repo)
        echo "=== Verify: $MAC_REPO vs github origin/$GIT_BRANCH ==="
        echo
        verify_repo
        echo
        if [ "$REPO_DIRTY" -ne 0 ] || [ "$REPO_BEHIND" -ne 0 ]; then exit 4; fi
        ;;

    --relocate)
        # The repo moved on this Mac, so its slug changed and claude --resume
        # can no longer see the old sessions. Rewrite the embedded paths and
        # fast-forward-merge them into the new slug. Safe to re-run: it is the
        # only way to sweep up a session that was still being written during
        # the move, since Claude Code resolves cwd once at startup.
        OLD_SLUG="$(slug "$OLD_REPO")"
        OLD_SESSIONS="$HOME/.claude/projects/$OLD_SLUG"

        echo "=== Relocate sessions (no SSH) ==="
        echo "    $OLD_REPO"
        echo "    ->  $MAC_REPO"
        echo

        [ -d "$OLD_SESSIONS" ] || { echo "FATAL: no sessions at $OLD_SESSIONS"; exit 1; }
        if [ "$OLD_SESSIONS" = "$MAC_SESSIONS" ]; then
            echo "FATAL: old and new paths give the same slug — nothing to relocate"
            exit 1
        fi

        echo "[1/2] rewrite  $OLD_REPO  ->  $MAC_REPO"
        rewrite_tree "$OLD_SESSIONS" "$STAGE" "$OLD_REPO" "$MAC_REPO"

        echo "[2/2] merge into $MAC_SESSIONS"
        merge_tree "$STAGE" "$MAC_SESSIONS"
        rm -rf "$STAGE"

        echo
        if [ "$DIVERGED" -gt 0 ]; then
            echo "WARNING: $DIVERGED file(s) diverged — nothing was overwritten."
            exit 3
        fi
        echo "Done. On the Mac:  cd $MAC_REPO && claude --resume"
        echo "The old slug dir is left intact as a full backup:"
        echo "    $OLD_SESSIONS"
        echo "Re-run this after ending any session that was live during the move."
        ;;

    --import-legacy)
        echo "=== Import legacy backup -> live Mac sessions (no SSH) ==="
        echo "    $LEGACY_DIR"
        echo "    ->  $MAC_SESSIONS"
        echo

        [ -d "$LEGACY_DIR" ] || { echo "FATAL: no legacy backup at $LEGACY_DIR"; exit 1; }

        echo "[1/2] rewrite  $GPU_REPO  ->  $MAC_REPO"
        rewrite_tree "$LEGACY_DIR" "$STAGE" "$GPU_REPO" "$MAC_REPO"

        echo "[2/2] merge into $MAC_SESSIONS"
        merge_tree "$STAGE" "$MAC_SESSIONS"
        rm -rf "$STAGE"

        echo
        if [ "$DIVERGED" -gt 0 ]; then
            echo "WARNING: $DIVERGED file(s) diverged — nothing was overwritten."
            exit 3
        fi
        echo "Done. On the Mac:  cd $MAC_REPO && claude --resume"
        echo "Once you are happy, the old folder can go:  rm -rf $LEGACY_BASE/.claude_sessions"
        ;;
esac
