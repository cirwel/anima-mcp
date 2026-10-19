#!/bin/bash
# Make the deployed checkout's Git ref and index describe the files deploy.sh
# just rsynced, and prove those files are exactly the named commit.
#
# rsync deliberately excludes .git, so update the checkout's ref and index
# separately. Refresh the configured remote-tracking refs before fetching the
# exact SHA; otherwise a normal main deployment can look locally "ahead" only
# because origin/main is a stale cache. Mixed reset writes no working-tree
# files: it makes Git describe the files that were just copied, then the clean
# check proves they really do match the named commit.
#
# rsync also runs without --delete, because the checkout holds untracked
# runtime state beside the code, so a file the deployed history deleted is left
# behind. Such a leftover is removed only when it is byte-identical to the
# version its deleting commit removed: Git still holds every byte, and nothing
# written on the Pi is lost. Any other untracked or modified file stays, fails
# the clean check, and is named on stderr.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DEPLOYED_REF" >&2
    exit 2
fi
DEPLOYED_REF="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

git -C "$REPO_ROOT" fetch --quiet --no-tags origin
git -C "$REPO_ROOT" fetch --quiet --no-tags origin "$DEPLOYED_REF"
git -C "$REPO_ROOT" reset --mixed "$DEPLOYED_REF" >/dev/null
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$DEPLOYED_REF"

while IFS= read -r -d '' path; do
    deleted_by="$(git -C "$REPO_ROOT" log -1 --format=%H --no-renames --diff-filter=D "$DEPLOYED_REF" -- "$path")"
    [ -n "$deleted_by" ] || continue
    [ "$(git -C "$REPO_ROOT" hash-object -- "$path")" = \
        "$(git -C "$REPO_ROOT" rev-parse --verify --quiet "$deleted_by^:$path")" ] || continue
    rm -- "$REPO_ROOT/$path"
    echo "  Removed $path (deleted by ${deleted_by:0:7})"
done < <(git -C "$REPO_ROOT" ls-files -z --others --exclude-standard)

LEFT="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
if [ -n "$LEFT" ] || ! git -C "$REPO_ROOT" diff-index --quiet "$DEPLOYED_REF" --; then
    echo "$LEFT" >&2
    exit 1
fi
