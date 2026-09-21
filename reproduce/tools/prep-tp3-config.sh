#!/usr/bin/env bash
# Prepare TP3 virtual-heads config (hardlink tree) from Pollard pack.
# Usage (on a host with the models directory mounted):
#   bash tools/prep-tp3-config.sh
# Shard presence is checked here; hash-level verification of the download
# is available separately: python3 check-model-manifest.py <SRC> (sizes and
# SHA-256 for all LFS files, including every shard).
set -euo pipefail

EXPECTED_REVISION=f129e31a81e1337aa33e129e2d847fc7e37c8733
SRC=${SRC:-/models/safetensors/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard}
# hoist if hfdownloader nested
if [[ -f $SRC/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard/config.json ]]; then
  SRC=$SRC/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard
fi
DST=${DST:-/models/safetensors/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard-TP3}

# refuse destructive or nonsensical DST values
[[ -n $DST ]] || { echo "DST must not be empty"; exit 2; }
[[ ${DST%/} != "/" ]] || { echo "DST must not be /"; exit 2; }
[[ $DST != "$SRC" && $DST != "$SRC"/* ]] || { echo "DST must differ from SRC and not sit inside it"; exit 2; }
[[ ! -e $DST || -d $DST ]] || { echo "DST exists and is not a directory: $DST"; exit 2; }

cd "$SRC"

# shard presence via glob array (no ls parsing)
shards=( model-000??-of-00048.safetensors )
n=${#shards[@]}
echo "shards present: $n/48"
[[ "$n" == 48 ]] || { echo "NOT COMPLETE"; exit 2; }

rm -rf "$DST"
mkdir -p "$DST"

# hardlink every entry; glob arrays, not ls parsing
shopt -s nullglob dotglob
entries=( * )
shopt -u dotglob
for f in "${entries[@]}"; do
  [[ "$f" == ".cache" ]] && continue
  cp -al "$f" "$DST/$f"
done
rm -f "$DST/config.json"

python3 - <<PY
import json
src = "$SRC/config.json"
dst = "$DST/config.json"
c = json.load(open(src))
tc = c["text_config"]
assert tc["num_attention_heads"] == 64 and tc["o_groups"] == 8, (tc["num_attention_heads"], tc.get("o_groups"))
tc["num_attention_heads"] = 72
tc["o_groups"] = 9
src_vh = {"num_attention_heads": 64, "o_groups": 8}
tc["virtual_heads_from"] = src_vh
c["virtual_heads_from"] = src_vh
c["kai_tp3_virtual_heads"] = "64 real heads / 8 o_groups padded to 72 / 9 for TP3"
c["source_revision"] = "$EXPECTED_REVISION"
json.dump(c, open(dst, "w"), indent=1)
print("wrote", dst, "heads", tc["num_attention_heads"], "o_groups", tc["o_groups"])
PY
echo "TP3 prep done at $DST (expected revision $EXPECTED_REVISION)"
