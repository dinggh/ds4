#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${QWEN35_GGUF:-/Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf}

if [ ! -x "$ROOT/ds4-bench" ]; then
  echo "missing $ROOT/ds4-bench; build with make ds4-bench first" >&2
  exit 2
fi

if [ ! -f "$MODEL" ]; then
  echo "missing Qwen GGUF: $MODEL" >&2
  exit 2
fi

prompt=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-bench-prompt.XXXXXX")
out=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-bench-out.XXXXXX")
trap 'rm -f "$prompt" "$out"' EXIT HUP INT TERM

i=0
while [ "$i" -lt 24 ]; do
  printf 'Qwen benchmark paragraph %d: recurrent GDN full attention quantized decode snapshot restore continuation.\n' "$i" >>"$prompt"
  i=$((i + 1))
done

"$ROOT/ds4-bench" --metal -m "$MODEL" \
  --prompt-file "$prompt" \
  --ctx-start 8 --ctx-max 12 --step-incr 4 \
  --ctx-alloc 64 --gen-tokens 2 >"$out"

grep -q '^ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes$' "$out"
grep -Eq '^8,8,[0-9.]+,2,[0-9.]+,[1-9][0-9]*$' "$out"
grep -Eq '^12,4,[0-9.]+,2,[0-9.]+,[1-9][0-9]*$' "$out"

echo "qwen35 bench snapshot smoke passed: $MODEL"
