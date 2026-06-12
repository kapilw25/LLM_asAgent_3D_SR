#!/bin/bash
: '
=============================================================================
Mac <-> GPU Sync — Claude sessions, .env, git_push.sh
=============================================================================

Runs ON THE MAC. Uses scp/ssh against the SSH alias defined in ~/.ssh/config.
Bypasses GitHub entirely so secrets and session JSONLs never leave your machines.

Required SSH alias (Mac ~/.ssh/config):
    Host vast_RTXpro6000_96GB
        HostName <ip>
        User root
        Port <port>
        IdentityFile ~/.ssh/id_ed25519

Usage:
    bash claude_session.sh --upload --host vast_RTXpro4000_24GB     # push .env + git_push.sh + sessions
        # Mac -> GPU. Run on a fresh GPU instance.
        # Pushes:
        #   .env         -> /workspace/factorjepa/.env
        #   git_push.sh  -> /workspace/factorjepa/git_push.sh
        #   sessions     -> ~/.claude/projects/-workspace-factorjepa/
        # On GPU after this, run: claude --resume

    bash claude_session.sh --download --host vast_RTXpro4000_24GB   # back up sessions before destroy
        # GPU -> Mac. Run BEFORE destroying the GPU instance.
        # Pulls:
        #   GPU ~/.claude/projects/-workspace-factorjepa/
        #     -> Mac $MAC_BASE/.claude_sessions/projects/-workspace-factorjepa/

    --host <ssh_alias>   (optional) SSH alias from ~/.ssh/config to target.
        # Default: vast_RTXpro6000_96GB
        # e.g.  bash claude_session.sh --upload --host vast_RTXpro4000_24GB

Notes:
    - Sessions are keyed by absolute path. GPU project MUST live at
      /workspace/factorjepa for claude --resume to find them.
    - SSH alias IP/port changes per instance — update ~/.ssh/config each time.
    - .env and git_push.sh are uploaded one-way (Mac -> GPU); they originate
      on the Mac and are never pulled back.

=============================================================================
'

set -euo pipefail

# === Configurable ===
SSH_HOST="vast_RTXpro6000_96GB"   # default — override per-run with --host <ssh_alias>

usage() {
    echo "Usage: ./claude_session.sh --upload|--download [--host <ssh_alias>]"
    echo "  e.g. ./claude_session.sh --upload --host vast_RTXpro4000_24GB"
}

MODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --upload|--download)
            MODE="$1" ;;
        --host)
            shift
            [ $# -gt 0 ] || { echo "Error: --host requires an SSH alias (from ~/.ssh/config)"; usage; exit 1; }
            SSH_HOST="$1" ;;
        *)
            echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
    shift
done

if [ -z "$MODE" ]; then
    echo "Error: --upload or --download required"
    usage
    exit 1
fi
MAC_BASE="/Users/kapilwanaskar/Downloads/research_projects/factorjepa"
GPU_REPO="/workspace/factorjepa"
PROJECT_SLUG="-workspace-factorjepa"

# === Derived ===
MAC_ENV="$MAC_BASE/.env"
MAC_GITPUSH="$MAC_BASE/git_push.sh"
MAC_SESSIONS_DIR="$MAC_BASE/.claude_sessions/projects"
MAC_SESSIONS="$MAC_SESSIONS_DIR/$PROJECT_SLUG"

# Light guard — most macs aren't named "Linux"; this protects against accidental
# runs on the GPU where outbound SSH to $SSH_HOST will not work.
if [ "$(uname)" != "Darwin" ]; then
    echo "Warning: this script is meant to run on the Mac, not on $(uname). Continuing anyway."
fi

case "$MODE" in
    --upload)
        echo "=== Upload: Mac -> GPU ($SSH_HOST) ==="

        [ -f "$MAC_ENV" ]     || { echo "Error: $MAC_ENV missing on Mac"; exit 1; }
        [ -f "$MAC_GITPUSH" ] || { echo "Error: $MAC_GITPUSH missing on Mac"; exit 1; }

        ssh "$SSH_HOST" "test -d $GPU_REPO" \
            || { echo "Error: $GPU_REPO not present on GPU — clone factorjepa there first"; exit 1; }

        echo "[1/3] .env -> $GPU_REPO/.env"
        scp "$MAC_ENV" "$SSH_HOST:$GPU_REPO/.env"

        echo "[2/3] git_push.sh -> $GPU_REPO/git_push.sh"
        scp "$MAC_GITPUSH" "$SSH_HOST:$GPU_REPO/git_push.sh"
        ssh "$SSH_HOST" "chmod +x $GPU_REPO/git_push.sh"

        echo "[3/3] sessions -> ~/.claude/projects/$PROJECT_SLUG"
        if [ -d "$MAC_SESSIONS" ]; then
            ssh "$SSH_HOST" "mkdir -p ~/.claude/projects && rm -rf ~/.claude/projects/$PROJECT_SLUG"
            scp -r "$MAC_SESSIONS" "$SSH_HOST:.claude/projects/"
            echo "Sessions restored — on GPU run: claude --resume"
        else
            echo "No Mac backup at $MAC_SESSIONS — skipping (first run on this Mac?)"
        fi

        echo ""
        echo "Done. On GPU:  cd $GPU_REPO && claude --resume"
        ;;

    --download)
        echo "=== Download: GPU ($SSH_HOST) -> Mac ==="

        ssh "$SSH_HOST" "test -d ~/.claude/projects/$PROJECT_SLUG" \
            || { echo "Error: GPU has no sessions at ~/.claude/projects/$PROJECT_SLUG"; exit 1; }

        mkdir -p "$MAC_SESSIONS_DIR"
        rm -rf "$MAC_SESSIONS"
        scp -r "$SSH_HOST:.claude/projects/$PROJECT_SLUG" "$MAC_SESSIONS_DIR/"

        echo "Backed up to $MAC_SESSIONS"
        du -sh "$MAC_SESSIONS"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./claude_session.sh --upload | --download"
        exit 1
        ;;
esac
