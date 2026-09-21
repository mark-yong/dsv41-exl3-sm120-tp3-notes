#!/usr/bin/env bash
# Prepare TP3 virtual-heads config (hardlink tree) from Pollard pack.
# Usage (on a host with the models directory mounted):
#   bash tools/prep-tp3-config.sh
set -euo pipefail
SRC=${SRC:-/models/safetensors/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard}
# hoist if hfdownloader nested
if [[ -f $SRC/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard/config.json ]]; then
  SRC=$SRC/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard
fi
DST=${DST:-/models/safetensors/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard-TP3}
cd "$SRC"
n=$(ls | grep -cE '^model-000[0-9]+-of-00048\.safetensors$' || true)
echo "shards present: $n/48"
[[ "$n" == "48" ]] || { echo "NOT COMPLETE"; exit 2; }
rm -rf "$DST"
mkdir -p "$DST"
for f in $(ls -A | grep -v '^\.cache$'); do
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
json.dump(c, open(dst, "w"), indent=1)
print("wrote", dst, "heads", tc["num_attention_heads"], "o_groups", tc["o_groups"])
PY
echo "TP3 prep done at $DST"
