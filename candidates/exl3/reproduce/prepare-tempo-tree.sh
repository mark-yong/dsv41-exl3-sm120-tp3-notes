#!/usr/bin/env bash
# Clone the pinned jspark3-deepseek revision and apply the tempo-overlay
# files on top. The produced tree is ready for `bash build.sh`.
# Usage: bash prepare-tempo-tree.sh <dest-dir>
set -euo pipefail

REV=bb386d39098e582fa1446cb96260cdef1df794f9
UPSTREAM=https://github.com/jakejharris/jspark3-deepseek
DEST=${1:?usage: prepare-tempo-tree.sh <dest-dir>}
REPO_DIR=$(cd "$(dirname "$0")" && pwd)

[[ ! -e $DEST ]] || { echo "destination exists: $DEST"; exit 2; }
git clone -q "$UPSTREAM" "$DEST"
git -C "$DEST" checkout -q "$REV"
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$REV" ]] || { echo "revision mismatch after checkout"; exit 1; }

cp "$REPO_DIR/tempo-overlay/stage.py" "$DEST/build/stage.py"
cp "$REPO_DIR/tempo-overlay/Dockerfile" "$DEST/Dockerfile"
cp "$REPO_DIR/tempo-overlay/build.sh" "$DEST/build.sh"
cp "$REPO_DIR/tempo-overlay/sources-amd64.json" "$DEST/release/sources-amd64.json"

# keep build scratch inside the tree, out of the way
export WORK="${WORK:-$DEST/.build}"

echo "prepared $DEST at $REV (overlay applied)"
echo "next: cd $DEST && bash build.sh   # build.sh re-verifies the tree and the base digest"
