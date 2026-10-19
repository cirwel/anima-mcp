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
# behind. Such a leftover is removed only when it is a regular file
# byte-identical to some version of the same path in the deployed history: Git
# still holds every byte, and nothing written on the Pi is lost. A symlink, or
# any other untracked or modified file, stays, fails the clean check, and is
# named on stderr.

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
    file="$REPO_ROOT/$path"
    if [ ! -f "$file" ] || [ -L "$file" ]; then
        continue
    fi
    blob="$(git -C "$REPO_ROOT" hash-object -- "$path")" || continue
    # awk reads the whole log, so pipefail cannot see an early-exit SIGPIPE.
    # Appending "" compares hashes as strings; awk compares 0e12... as numbers.
    git -C "$REPO_ROOT" log --raw --no-abbrev --no-renames --format= "$DEPLOYED_REF" -- ":(literal)$path" \
        | awk -v blob="$blob" '($3 "") == (blob "") || ($4 "") == (blob "") { found = 1 } END { exit !found }' \
        || continue
    rm -- "$file"
    echo "  Removed leftover $path (an old version Git still holds)"
done < <(git -C "$REPO_ROOT" ls-files -z --others --exclude-standard)

LEFT="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
if [ -n "$LEFT" ]; then
    echo "$LEFT" >&2
    exit 1
fi
git -C "$REPO_ROOT" diff-index --quiet "$DEPLOYED_REF" --
