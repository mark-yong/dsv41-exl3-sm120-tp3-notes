#!/usr/bin/env python3
"""Extract llm-inference-bench prefill dict into a one-line-per-ctx summary."""
import json
import sys

path, out = sys.argv[1], sys.argv[2]
d = json.load(open(path))
pref = d.get("prefill") or {}
lines = ["# Prefill summary (tok/s = prompt_tokens/TTFT)", f"source={path}"]
for ctx, row in sorted(pref.items(), key=lambda x: int(x[0])):
    srv = (row.get("server_validation") or {}).get("tok_per_sec")
    lines.append(
        f"ctx={ctx} tokens={row.get('prompt_tokens')} "
        f"client_tok_s={row.get('tok_per_sec')} "
        f"server_tok_s={srv} ttft_s={row.get('ttft_seconds')}"
    )
text = "\n".join(lines) + "\n"
open(out, "w").write(text)
print(text, end="")
