"""HF LFS bloat audit + super_squash_history dry-run.

Detects orphaned LFS blobs (old revisions of overwritten/renamed/deleted files
that still count toward your HF account quota) and projects how much space
`HfApi.super_squash_history()` would reclaim. DEFAULT IS DRY-RUN — must pass
both `--execute` AND `--confirm "<phrase>"` to actually squash.

iter16 (2026-05-22): built after the audit found 870 GB of "missing" account
quota (1.26 TB dashboard - 386.5 GB tree-sum = 870 GB orphaned LFS blobs from
iter11→iter16 checkpoint churn). HfApi.list_repo_tree() reports only CURRENT-
revision file sizes; HF dashboard shows the union over ALL historical commits
(including orphaned LFS blobs after delete/overwrite/rename). super_squash_
history collapses the branch into a single commit, making old LFS blobs
unreferenced → HF's background GC reclaims them within minutes-to-hours.

USAGE:
    # Audit only (read-only, safe — DEFAULT, account-wide):
    venv_walkindia/bin/python -u tests/lfs_squash_audit.py --user anonymousML123

    # Audit + squash a SPECIFIC subset of repos (DESTRUCTIVE):
    venv_walkindia/bin/python -u tests/lfs_squash_audit.py --user anonymousML123 \\
        --repos anonymousML123/factorjepa-outputs anonymousML123/walkindia-200k \\
        --execute --confirm "I understand commit history will be erased"

    # Audit + squash EVERY repo on the account (DESTRUCTIVE, account-wide):
    venv_walkindia/bin/python -u tests/lfs_squash_audit.py --user anonymousML123 \\
        --all-account \\
        --execute --confirm "I understand commit history will be erased"

Audit scope is ALWAYS the whole account (every dataset + model under --user).
The --repos / --all-account flags only narrow what gets SQUASHED, not what
gets audited. Reads HF_TOKEN from .env via python-dotenv.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

CONFIRM_PHRASE = "I understand commit history will be erased"


def load_token() -> str:
    """Load HF_TOKEN from .env. FAIL LOUD if missing."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("FATAL: python-dotenv not installed; run `pip install python-dotenv`",
              file=sys.stderr)
        sys.exit(1)
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        print(f"FATAL: HF_TOKEN not in {repo_root / '.env'}", file=sys.stderr)
        sys.exit(1)
    return token


def repo_current_size(api: HfApi, repo_id: str, repo_type: str) -> tuple:
    """Sum CURRENT-revision file sizes via list_repo_tree. Returns (bytes, n_files, err)."""
    total = 0
    n_files = 0
    try:
        for item in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True):
            if isinstance(item, RepoFile):
                total += getattr(item, "size", 0) or 0
                n_files += 1
        return total, n_files, None
    except Exception as e:
        return 0, 0, f"{type(e).__name__}: {e}"


def repo_commit_count(api: HfApi, repo_id: str, repo_type: str) -> int:
    """Count commits on main — proxy for LFS churn. Returns 0 on error (non-fatal)."""
    try:
        return len(list(api.list_repo_commits(repo_id, repo_type=repo_type)))
    except Exception:
        return 0


def list_user_repos(api: HfApi, user: str) -> list:
    """Enumerate all dataset+model repos owned by user. Skips spaces (no LFS quota)."""
    repos = []
    for r in api.list_datasets(author=user):
        repos.append((r.id, "dataset"))
    for r in api.list_models(author=user):
        repos.append((r.id, "model"))
    return repos


def print_audit_table(rows: list) -> int:
    """Print box-drawing audit table. Returns grand total bytes."""
    print()
    print("┌─────┬─────────────────────────────────────────────────────────────┬─────────┬───────┬─────────┬─────────────┐")
    print("│  #  │ Repo                                                         │ Type    │ Files │ Commits │ Current GB  │")
    print("├─────┼─────────────────────────────────────────────────────────────┼─────────┼───────┼─────────┼─────────────┤")
    grand = 0
    for i, r in enumerate(rows, 1):
        grand += r["size_bytes"]
        print(f"│ {i:>3} │ {r['id']:60s} │ {r['type']:7s} │ {r['files']:>5d} │ {r['commits']:>7d} │ {r['size_gb']:>11.2f} │")
    print("└─────┴─────────────────────────────────────────────────────────────┴─────────┴───────┴─────────┴─────────────┘")
    print(f"\nGRAND TOTAL (current revision): {grand/1e9:.2f} GB = {grand/1e12:.3f} TB")
    return grand


def print_bloat_projection(grand_current_bytes: int, dashboard_tb: float) -> None:
    """Compute and print the implied LFS-bloat reclaim."""
    dashboard_gb = dashboard_tb * 1024.0      # TB → GB (binary), but HF uses decimal — see note
    # HF reports storage in decimal GB (1 GB = 10^9 bytes), not binary GiB.
    dashboard_gb = dashboard_tb * 1000.0
    current_gb = grand_current_bytes / 1e9
    bloat_gb = max(0.0, dashboard_gb - current_gb)
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ 📊 LFS BLOAT INFERENCE                                                       │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ HF account quota (dashboard input):  {dashboard_gb:>10.1f} GB ({dashboard_tb:.2f} TB)        │")
    print(f"│ Sum of CURRENT-revision file sizes:  {current_gb:>10.1f} GB                       │")
    print(f"│ ─────────────────────────────────────────────────────────────────────────  │")
    print(f"│ Implied orphaned LFS blobs (reclaim): {bloat_gb:>10.1f} GB ({bloat_gb/1000:.2f} TB)        │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print("\nNote: tree-sum counts files in main HEAD only. Bloat = old LFS revisions")
    print("retained by HF after overwrite/rename/delete operations across commits.")


def squash_candidates(rows: list) -> list:
    """Rank repos by churn-proxy (commits × current_size). Highest churn → biggest reclaim."""
    return sorted(rows, key=lambda r: r["commits"] * r["size_gb"], reverse=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True,
                   help="HF username (lists all their datasets + models). FAIL LOUD if missing.")
    p.add_argument("--dashboard-tb", type=float, default=1.26,
                   help="HF dashboard reported storage in TB (decimal). Used to compute "
                        "implied LFS bloat = dashboard_GB - tree_sum_GB. "
                        "Default 1.26 TB (the value the user reported on 2026-05-22).")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--repos", nargs="+",
                       help="Specific repo IDs to squash. Mutually exclusive with --all-account.")
    scope.add_argument("--all-account", action="store_true",
                       help="Squash EVERY enumerated repo under --user (account-wide). "
                            "Mutually exclusive with --repos.")
    p.add_argument("--execute", action="store_true",
                   help="Actually call super_squash_history(). DESTRUCTIVE — commit history "
                        "is erased. Requires --confirm.")
    p.add_argument("--confirm", default="",
                   help=f"Required when --execute is set. Must equal: {CONFIRM_PHRASE!r}")
    args = p.parse_args()

    if args.execute and args.confirm != CONFIRM_PHRASE:
        print(f"FATAL: --execute requires --confirm {CONFIRM_PHRASE!r}", file=sys.stderr)
        print(f"  You passed: {args.confirm!r}", file=sys.stderr)
        return 2
    if args.execute and not (args.repos or args.all_account):
        print(f"FATAL: --execute requires either --repos REPO1 REPO2 ... or --all-account",
              file=sys.stderr)
        return 2

    token = load_token()
    api = HfApi(token=token)

    print(f"╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║ 🔍 HF LFS BLOAT AUDIT — user={args.user}                                  ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝")

    repos = list_user_repos(api, args.user)
    if not repos:
        print(f"FATAL: no datasets or models found for user={args.user}", file=sys.stderr)
        return 3
    print(f"\nEnumerated {len(repos)} repos (datasets + models).")

    rows = []
    for repo_id, repo_type in repos:
        size, n_files, err = repo_current_size(api, repo_id, repo_type)
        if err:
            print(f"  WARN: skipping {repo_id}: {err}")
            continue
        n_commits = repo_commit_count(api, repo_id, repo_type)
        rows.append({
            "id":         repo_id,
            "type":       repo_type,
            "files":      n_files,
            "commits":    n_commits,
            "size_bytes": size,
            "size_gb":    size / 1e9,
        })

    if not rows:
        print(f"FATAL: every repo errored out during size-fetch", file=sys.stderr)
        return 4

    grand_current = print_audit_table(rows)
    print_bloat_projection(grand_current, args.dashboard_tb)

    ranked = squash_candidates(rows)
    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🎯 SQUASH CANDIDATES (ranked by churn-proxy = commits × current_GB)                       │")
    print(f"├──────────────────────────────────────────────────────────────────────────────────────────┤")
    for r in ranked[:10]:
        churn = r["commits"] * r["size_gb"]
        print(f"│ {r['id']:60s}  commits={r['commits']:>4d}  cur={r['size_gb']:>7.2f} GB  churn={churn:>9.1f} │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────┘")

    if not (args.repos or args.all_account):
        print(f"\n✋ AUDIT-ONLY mode (neither --repos nor --all-account passed).")
        print(f"   Re-run with one of:")
        print(f"     --repos REPO1 REPO2 ...   (target a subset)")
        print(f"     --all-account             (target every repo above)")
        print(f"   Add --execute --confirm \"{CONFIRM_PHRASE}\" to actually squash.")
        return 0

    # Build squash plan
    by_id = {r["id"]: r for r in rows}
    plan = []
    if args.all_account:
        plan = list(rows)            # every enumerated repo
        print(f"\n📌 --all-account: targeting all {len(plan)} enumerated repos")
    else:
        for repo_id in args.repos:
            if repo_id not in by_id:
                print(f"  ❌ --repos {repo_id} not in enumerated list for user={args.user} — skipping")
                continue
            plan.append(by_id[repo_id])

    if not plan:
        print(f"FATAL: no valid repos in --repos to act on", file=sys.stderr)
        return 5

    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 📋 SQUASH PLAN                                                                              │")
    print(f"├──────────────────────────────────────────────────────────────────────────────────────────┤")
    for r in plan:
        print(f"│ → {r['id']} ({r['type']})  {r['commits']} commits → 1 commit, "
              f"{r['size_gb']:.2f} GB CURRENT preserved")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────┘")

    if not args.execute:
        print(f"\n✋ DRY-RUN — no mutation. Re-run with --execute --confirm \"{CONFIRM_PHRASE}\"")
        return 0

    print(f"\n🔥 EXECUTING super_squash_history (irreversible)")
    failures = []
    for r in plan:
        print(f"  ⚙️  squashing {r['id']} ({r['type']}) ...", end="", flush=True)
        try:
            api.super_squash_history(repo_id=r["id"], repo_type=r["type"],
                                      commit_message=f"super_squash_history (iter16 LFS cleanup, "
                                                     f"was {r['commits']} commits / "
                                                     f"{r['size_gb']:.2f} GB current)")
            print(" ✅")
        except Exception as e:
            print(f" ❌ {type(e).__name__}: {e}")
            failures.append((r["id"], str(e)))

    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🔁 POST-SQUASH VERIFICATION                                                                 │")
    print(f"├──────────────────────────────────────────────────────────────────────────────────────────┤")
    new_grand = 0
    for r in plan:
        size, n_files, err = repo_current_size(api, r["id"], r["type"])
        if err:
            print(f"│ ⚠️  {r['id']:60s}  re-fetch failed: {err}")
            continue
        new_grand += size
        print(f"│ {r['id']:60s}  was {r['size_gb']:>7.2f} GB → now {size/1e9:>7.2f} GB │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────┘")
    print(f"\nNote: HF dashboard reflects LFS-GC asynchronously (minutes to hours). The CURRENT-")
    print(f"revision file sums should be unchanged; quota recovery shows up on the dashboard after GC.")

    if failures:
        print(f"\n{len(failures)} repo(s) failed to squash:")
        for repo_id, err in failures:
            print(f"  ❌ {repo_id}: {err}")
        return 6

    return 0


if __name__ == "__main__":
    sys.exit(main())
