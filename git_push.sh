#!/bin/bash
: '
=============================================================================
Git Push Script - Multi-Account Support (CODE PUSH ONLY)
=============================================================================

Pushes code to GitHub only. The HF outputs-upload step was REMOVED (2026-05-30) —
this script no longer touches HuggingFace, so a forgotten flag can never trigger a
slow/destructive HF sync. Credentials (per-account GitHub PATs) are loaded from the
gitignored .env (GITHUB_TOKEN_KAPIL / GITHUB_TOKEN_GAYTRI) — never hardcoded here.

Usage:
    chmod +x git_push.sh           # make it executable (once)
    bash git_push.sh "updating"    # commit + push code (no flags — --code-only was removed)

To back up outputs/ to HF MANUALLY (separate, opt-in — no longer part of this script):
    python -u src/utils/hf_outputs.py upload outputs 2>&1 | tee logs/hf_upload_outputs_$(date +%Y%m%d_%H%M%S).log

Example:
    % ./git_push.sh "testing git push"
    Select account:
    1) Kapil
    2) Gaytri
    Enter choice [1/2]: 1
    ...
    Pushed as Kapil
    Done (code push only)

=============================================================================
'

MESSAGE=""

for arg in "$@"; do
    case "$arg" in
        --*) echo "Error: flags removed — this script is code-push only (no --code-only). Pass just the commit message."; exit 1 ;;
        *) MESSAGE="$arg" ;;
    esac
done

if [ -z "$MESSAGE" ]; then
    echo "Error: Commit message required"
    echo "Usage: ./git_push.sh \"commit message\""
    exit 1
fi

# Load credentials from .env (gitignored) — tokens NEVER live in this tracked script.
ENV_FILE="$(dirname "$0")/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "FATAL: $ENV_FILE not found (needs GITHUB_TOKEN_KAPIL / GITHUB_TOKEN_GAYTRI)"
    exit 1
fi
set -a; . "$ENV_FILE"; set +a

echo "Select account:"
echo "1) Kapil"
echo "2) Gaytri"
read -p "Enter choice [1/2]: " CHOICE

if [ "$CHOICE" = "2" ]; then
    git config user.name "GaytriJena"
    git config user.email "gaytrijena2000@gmail.com"
    USERNAME="GaytriJena"
    TOKEN="$GITHUB_TOKEN_GAYTRI"
    ACCOUNT="Gaytri"
else
    git config user.name "kapilw25"
    git config user.email "kapilw25@gmail.com"
    USERNAME="kapilw25"
    TOKEN="$GITHUB_TOKEN_KAPIL"
    ACCOUNT="Kapil"
fi

if [ -z "$TOKEN" ]; then
    echo "FATAL: GitHub token for $ACCOUNT not set in $ENV_FILE"
    echo "  add GITHUB_TOKEN_KAPIL=<pat> and GITHUB_TOKEN_GAYTRI=<pat> there"
    exit 1
fi

git add .
git commit -m "$MESSAGE"

# Disable credential helper temporarily and use token directly
GIT_TERMINAL_PROMPT=0 git -c credential.helper= push https://${USERNAME}:${TOKEN}@github.com/kapilw25/factorjepa.git main

echo "Pushed as $ACCOUNT"

# HF upload removed (2026-05-30): this script ONLY pushes code — it never touches HuggingFace,
# so there is no longer any HF step to skip (the old --code-only flag was therefore removed).
# HF backup is now a separate, opt-in manual step (run the outputs-upload util directly).
echo "Done (code push only)"
