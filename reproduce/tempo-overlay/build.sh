#!/usr/bin/env bash
# Build the Tempo amd64/SM120 image. WORK holds pinned inputs, context, and receipts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TAG="${TAG:-dsv41-tempo-sm120-tp3:amd64}"
WORK="${WORK:-$(cd "$(dirname "$0")" && pwd)/.build}"
INPUTS="${INPUTS:-$WORK/inputs}"
CTX="$WORK/context"

mkdir -p "$INPUTS" "$WORK"

echo "==> Fetching pinned archives into $INPUTS"
ROOT="$ROOT" INPUTS="$INPUTS" python3 <<'PY'
import json, os, urllib.request, hashlib
from pathlib import Path

root = Path(os.environ["ROOT"])
pins = json.loads((root / "release/sources-amd64.json").read_text())
dest = Path(os.environ["INPUTS"])
dest.mkdir(parents=True, exist_ok=True)

def verify(path: Path, sha: str, nbytes):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            size += len(b)
            h.update(b)
    got = h.hexdigest()
    if nbytes is not None and size != nbytes:
        raise SystemExit(f"size mismatch {path.name}: {size} != {nbytes}")
    if got != sha:
        raise SystemExit(f"sha256 mismatch {path.name}: {got} != {sha}")

for pin in list(pins["archives"].values()) + [pins["cmake"]]:
    dst = dest / pin["file"]
    if dst.exists():
        print(f"rehash {pin['file']}", flush=True)
        verify(dst, pin["sha256"], pin.get("bytes"))
        continue
    tmp = dst.with_suffix(dst.suffix + ".incomplete")
    print(f"download {pin['file']}", flush=True)
    with urllib.request.urlopen(pin["url"], timeout=300) as src, tmp.open("wb") as out:
        while True:
            b = src.read(8 << 20)
            if not b:
                break
            out.write(b)
        out.flush()
        os.fsync(out.fileno())
    verify(tmp, pin["sha256"], pin.get("bytes"))
    os.replace(tmp, dst)

out_pins = {
    "schema_version": pins["schema_version"],
    "base": pins["base"],
    "flashinfer_version": pins["flashinfer_version"],
    "archives": pins["archives"],
    "cmake": pins["cmake"],
}
(dest / "pins.json").write_text(json.dumps(out_pins, indent=2) + "\n")
print("PASS: inputs verified", flush=True)
PY

echo "==> Assembling build context at $CTX"
rm -rf "$CTX"
mkdir -p "$CTX/recipe"
cp -a "$INPUTS" "$CTX/inputs"
cp -a "$ROOT/build" "$CTX/recipe/build"
cp -a "$ROOT/tools" "$CTX/recipe/tools"
cp -a "$ROOT/patches" "$CTX/recipe/patches"
cp -a "$ROOT/release" "$CTX/recipe/release"
cp -f "$ROOT/Dockerfile" "$CTX/Dockerfile"

SOURCES_SHA256="$(sha256sum "$ROOT/release/sources-amd64.json" | awk '{print $1}')"
echo "SOURCES_SHA256=$SOURCES_SHA256"

echo "==> docker pull base"
docker pull "vllm/vllm-openai@sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d"

echo "==> Tempo tree integrity receipt"
TEMPOR_ROOT="$(pwd)"
git rev-parse HEAD | grep -qx "$(python3 -c "import json;print(json.load(open('release/sources-amd64.json'))['tempo_git_rev'])")" \
  || { echo "FAIL: checkout is not the pinned Tempo revision"; exit 1; }
# The overlay files this repo replaces upstream are expected to differ from
# the clone; they must not make the tree look dirty. The build.sh / Dockerfile
# / stage.py / sources-amd64.json set IS the amd64 port under test.
OVERLAY_EXCLUDES=( Dockerfile build.sh build/stage.py release/sources-amd64.json )
for f in "${OVERLAY_EXCLUDES[@]}"; do git update-index --assume-unchanged "$f"; done
if ! git diff --quiet; then
  git update-index --no-assume-unchanged "${OVERLAY_EXCLUDES[@]}"
  echo "FAIL: working tree is dirty outside the overlay files; refusing to build from a modified checkout"
  git status --short
  exit 1
fi
git update-index --no-assume-unchanged "${OVERLAY_EXCLUDES[@]}"
find build tools patches release Dockerfile -type f -exec sha256sum {} + | sed "s|$TEMPOR_ROOT/||" | sort > tempo-tree.sha256
sha256sum tempo-tree.sha256

echo "==> docker build (legacy builder; MAX_JOBS=16 inside stages)"
export DOCKER_BUILDKIT=0
# Tempo builds with --network=none; this build keeps that after the base pull (archives are pre-staged in the context).
docker build \
  --network=none \
  --memory=96g \
  --memory-swap=96g \
  --build-arg "SOURCES_SHA256=$SOURCES_SHA256" \
  --tag "$TAG" \
  "$CTX"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$TAG")"
cat >"$WORK/image.json" <<EOF
{
  "tag": "$TAG",
  "image_id": "$IMAGE_ID",
  "sources_sha256": "$SOURCES_SHA256",
  "arch_list": "12.0a",
  "path": "tempo-amd64",
  "base": "vllm/vllm-openai:deepseekv41-flash-0909",
  "base_digest": "sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d"
}
EOF
echo "PASS image $IMAGE_ID"
cat "$WORK/image.json"
