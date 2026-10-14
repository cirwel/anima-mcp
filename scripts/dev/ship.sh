#!/usr/bin/env bash
# ship.sh — agent-friendly commit-and-deliver (fleet-portable core)
#
# Default delivery in `auto` mode follows the fleet routing contract and the
# 2026-10-07 operator delegation for non-unitares repos:
#   - Runtime code (src/anima_mcp/, anima_broker/, systemd/, pyproject.toml),
#     detached HEAD work, or feature-branch work → ready PR + auto-merge on
#     green CI (mints a fresh agent-prefixed branch when on main or detached).
#   - Everything else (docs, scripts, tests, lockfile) on `main` → direct
#     commit + push to origin/main.
#
# Pass `--deploy` (or `ANIMA_SHIP_DEPLOY=1`) to also deploy the landed `main`
# commit to Lumen's Pi via `./deploy.sh --host "${PI_HOST:-lumen}"`.
#
# Usage:
#   ./scripts/dev/ship.sh "commit message"
#   ./scripts/dev/ship.sh --deploy "commit message"
#   ./scripts/dev/ship.sh --stage-all "commit message"
#   ./scripts/dev/ship.sh --draft-pr "commit message"
#   ./scripts/dev/ship.sh --open-pr "commit message"
#   ./scripts/dev/ship.sh --auto-merge "commit message"
#   ./scripts/dev/ship.sh --direct "commit message"
#   ./scripts/dev/ship.sh --classify          # just print "runtime" or "other"
#   ./scripts/dev/ship.sh --plan "commit message"
#
# Requirements: staged changes (git add already done) unless --stage-all is
# used, gh CLI authed.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

RUNTIME_PATTERNS=(
    '^src/anima_mcp/'
    '^anima_broker/'
    '^systemd/'
    '^pyproject\.toml$'
)

classify_paths() {
    local files="$1"
    if [[ -z "$files" ]]; then
        echo "empty"; return
    fi
    while IFS= read -r f; do
        for pat in "${RUNTIME_PATTERNS[@]}"; do
            if [[ "$f" =~ $pat ]]; then
                echo "runtime"; return
            fi
        done
    done <<< "$files"
    echo "other"
}

classify() {
    local files; files=$(git diff --cached --name-only)
    classify_paths "$files"
}

worktree_changed_files() {
    {
        git diff --name-only
        git diff --cached --name-only
        git ls-files --others --exclude-standard
    } | sort -u
}

classify_worktree() {
    local files; files=$(worktree_changed_files)
    classify_paths "$files"
}

detect_agent_prefix() {
    local agent_prefix="${UNITARES_SHIP_AGENT:-}"
    if [[ -z "$agent_prefix" ]]; then
        if [[ -n "${CLAUDECODE:-}" ]]; then
            agent_prefix="claude"
        elif [[ -n "${ANTIGRAVITY_AGENT:-}" || -n "${GEMINI_CLI:-}" ]]; then
            agent_prefix="gemini"
        else
            agent_prefix="codex"
        fi
    fi
    echo "$agent_prefix"
}

usage() {
    cat >&2 <<'USAGE'
usage: ship.sh [--stage-all] [--deploy] [--draft-pr|--open-pr|--auto-merge|--direct] "commit message"
       ship.sh --classify
       ship.sh [--stage-all] [--deploy] --plan "commit message"

Modes:
  auto         runtime/detached/branch work opens a PR and merges on green CI;
               non-runtime work on main commits and pushes directly.
  --draft-pr   commit, push current/new branch, and open a draft PR
  --open-pr    commit, push current/new branch, and open a ready PR
  --auto-merge commit, push current/new branch, open a ready PR, and merge on green CI
  --direct     commit and push the current branch; refuses detached HEAD
  --deploy     after landing on origin/main, run ./deploy.sh --host lumen
  --stage-all  stage the full current worktree before classifying/committing.
               With --plan, previews that route without mutating the index.
USAGE
}

MODE="${UNITARES_SHIP_MODE:-auto}"
PLAN_ONLY=0
STAGE_ALL=0
DEPLOY_PI="${ANIMA_SHIP_DEPLOY:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --classify)
            classify
            exit 0
            ;;
        --plan|--dry-run)
            PLAN_ONLY=1
            shift
            ;;
        --stage-all|--all)
            STAGE_ALL=1
            shift
            ;;
        --deploy)
            DEPLOY_PI=1
            shift
            ;;
        --draft-pr|--draft)
            MODE="draft_pr"
            shift
            ;;
        --open-pr|--pr)
            MODE="open_pr"
            shift
            ;;
        --auto-merge)
            MODE="auto_merge"
            shift
            ;;
        --direct)
            MODE="direct"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

MESSAGE="${*:-}"
if [[ -z "$MESSAGE" ]]; then
    usage
    exit 2
fi

if [[ "$STAGE_ALL" == "1" && "$PLAN_ONLY" != "1" ]]; then
    echo "[ship] staging all worktree changes"
    git add -A
fi

if [[ "$STAGE_ALL" == "1" && "$PLAN_ONLY" == "1" ]]; then
    KIND=$(classify_worktree)
else
    KIND=$(classify)
fi
BRANCH=$(git branch --show-current)
HEAD_SHORT=$(git rev-parse --short HEAD)
DETACHED=0
if [[ -z "$BRANCH" ]]; then
    DETACHED=1
fi

normalize_mode() {
    case "$1" in
        auto|draft_pr|open_pr|auto_merge|direct)
            echo "$1" ;;
        draft-pr|draft)
            echo "draft_pr" ;;
        open-pr|pr)
            echo "open_pr" ;;
        auto-merge)
            echo "auto_merge" ;;
        *)
            echo "invalid" ;;
    esac
}

MODE=$(normalize_mode "$MODE")
if [[ "$MODE" == "invalid" ]]; then
    echo "invalid UNITARES_SHIP_MODE; expected auto, draft-pr, open-pr, auto-merge, or direct" >&2
    exit 2
fi

DELIVERY="$MODE"
FORCE_AUTO_BRANCH=0
if [[ "$MODE" == "auto" ]]; then
    if [[ "$KIND" == "other" && "$DETACHED" == "0" && ( "$BRANCH" == "main" || "$BRANCH" == "master" ) ]]; then
        DELIVERY="direct"
    else
        DELIVERY="auto_merge"
        if [[ "$KIND" == "runtime" || "$DETACHED" == "1" ]]; then
            FORCE_AUTO_BRANCH=1
        fi
    fi
elif [[ "$DETACHED" == "1" && "$MODE" != "direct" ]]; then
    FORCE_AUTO_BRANCH=1
fi

if [[ "$KIND" == "empty" ]]; then
    echo "nothing staged — stage files with 'git add' first" >&2
    exit 2
fi

if [[ "$DELIVERY" == "direct" && "$DETACHED" == "1" ]]; then
    echo "detached HEAD cannot use direct delivery; rerun with --auto-merge or create a branch" >&2
    exit 2
fi

if [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
    case "$DELIVERY" in
        draft_pr|open_pr|auto_merge)
            FORCE_AUTO_BRANCH=1 ;;
    esac
fi

if [[ "$PLAN_ONLY" == "1" ]]; then
    branch_label="${BRANCH:-"(detached)"}"
    echo "kind=$KIND"
    echo "branch=$branch_label"
    echo "head=$HEAD_SHORT"
    echo "mode=$MODE"
    echo "delivery=$DELIVERY"
    echo "force_auto_branch=$FORCE_AUTO_BRANCH"
    echo "stage_all=$STAGE_ALL"
    echo "deploy_pi=$DEPLOY_PI"
    echo "agent_prefix=$(detect_agent_prefix)"
    exit 0
fi

if [[ "$KIND" != "runtime" && "$KIND" != "other" ]]; then
    echo "unknown staged-change classification: $KIND" >&2
    exit 2
fi

create_auto_branch_if_needed() {
    if [[ "$FORCE_AUTO_BRANCH" != "1" ]]; then
        return 0
    fi

    local slug
    slug=$(printf '%s' "$MESSAGE" | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-' | cut -c1-40)
    if [[ -z "$slug" ]]; then
        slug="change"
    fi

    local agent_prefix
    agent_prefix=$(detect_agent_prefix)

    local new_branch="${agent_prefix}/auto/$(date +%Y%m%d-%H%M%S)-${slug}"
    echo "[ship] creating PR branch: $new_branch"
    git checkout -b "$new_branch"
    BRANCH="$new_branch"
}

merge_pr_when_green() {
    local pr_url="$1"
    if gh pr merge --auto --squash "$pr_url" 2>/dev/null; then
        echo "[ship] auto-merge armed on $pr_url"
        if [[ "$DEPLOY_PI" == "1" ]]; then
            echo "[ship] waiting for CI checks to pass before Pi deploy..."
            gh pr checks "$pr_url" --watch --interval 10
        fi
        return 0
    fi

    echo "[ship] base branch has no required status-check protection for --auto; waiting for CI checks before squash-merging..."
    # Give GitHub Actions a few seconds to register the check suite on the fresh push.
    local attempt
    for attempt in 1 2 3 4 5 6; do
        if gh pr checks "$pr_url" >/dev/null 2>&1; then
            break
        fi
        sleep 3
    done
    if gh pr checks "$pr_url" --watch --interval 10; then
        gh pr merge --squash "$pr_url"
        echo "[ship] merged $pr_url after green CI"
    else
        echo "[ship] CI checks failed on $pr_url; leaving PR open without merging" >&2
        return 1
    fi
}

create_or_show_pr() {
    local pr_kind="$1"
    local pr_title pr_body pr_url existing_url

    # GitHub caps PR titles at 256 chars; use only the first line so
    # multi-line commit messages don't fail PR creation.
    pr_title=$(printf '%s\n' "$MESSAGE" | head -n1)

    existing_url=$(gh pr view --json url --jq .url 2>/dev/null || true)
    if [[ -n "$existing_url" ]]; then
        echo "[ship] PR already exists for $BRANCH"
        pr_url="$existing_url"
        echo "$pr_url"
    else
        case "$pr_kind" in
            draft_pr)
                pr_body="Auto-shipped by ship.sh as a draft PR. Local work is visible; mark ready after validation/review."
                pr_url=$(gh pr create --draft --title "$pr_title" --body "$pr_body")
                ;;
            open_pr)
                pr_body="Auto-shipped by ship.sh as an open PR. CI gate applies."
                pr_url=$(gh pr create --title "$pr_title" --body "$pr_body")
                ;;
            auto_merge)
                pr_body="Auto-shipped by ship.sh. Auto-merge requested; CI gate applies."
                pr_url=$(gh pr create --title "$pr_title" --body "$pr_body")
                ;;
            *)
                echo "internal error: unknown PR kind $pr_kind" >&2
                exit 2
                ;;
        esac
        echo "$pr_url"
    fi

    if [[ "$pr_kind" == "auto_merge" ]]; then
        merge_pr_when_green "$pr_url"
    fi
}

deploy_to_pi_if_requested() {
    if [[ "$DEPLOY_PI" != "1" ]]; then
        return 0
    fi
    local pi_host="${PI_HOST:-lumen}"
    echo "[ship] syncing checkout to origin/main and deploying to Pi ($pi_host)..."
    git fetch origin main --quiet
    if [[ "$DETACHED" == "1" ]]; then
        git checkout --detach origin/main --quiet
    else
        git checkout --detach origin/main --quiet
    fi
    ./deploy.sh --host "$pi_host"
}

case "$DELIVERY" in
    direct)
        echo "[ship] direct path → commit + push on $BRANCH"
        git commit -m "$MESSAGE"
        # Push to the same-name branch on origin, not whatever upstream tracks.
        git push -u origin "HEAD:$BRANCH"
        deploy_to_pi_if_requested
        ;;
    draft_pr|open_pr|auto_merge)
        create_auto_branch_if_needed
        echo "[ship] PR path → $BRANCH ($DELIVERY)"
        git commit -m "$MESSAGE"
        git push -u origin "$BRANCH"
        create_or_show_pr "$DELIVERY"
        if [[ "$DELIVERY" == "auto_merge" ]]; then
            deploy_to_pi_if_requested
        fi
        ;;
    *)
        echo "internal error: unknown delivery path $DELIVERY" >&2
        exit 2
        ;;
esac
