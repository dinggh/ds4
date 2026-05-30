#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_DIR=${QWEN35_VARIANTS_DIR:-/Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF}

models="
Qwen3.5-0.8B-BF16.gguf
Qwen3.5-0.8B-IQ4_NL.gguf
Qwen3.5-0.8B-IQ4_XS.gguf
Qwen3.5-0.8B-Q3_K_M.gguf
Qwen3.5-0.8B-Q3_K_S.gguf
Qwen3.5-0.8B-Q4_0.gguf
Qwen3.5-0.8B-Q4_1.gguf
Qwen3.5-0.8B-Q4_K_M.gguf
Qwen3.5-0.8B-Q4_K_S.gguf
Qwen3.5-0.8B-Q5_K_M.gguf
Qwen3.5-0.8B-Q5_K_S.gguf
Qwen3.5-0.8B-Q6_K.gguf
Qwen3.5-0.8B-Q8_0.gguf
Qwen3.5-0.8B-UD-IQ2_M.gguf
Qwen3.5-0.8B-UD-IQ3_XXS.gguf
Qwen3.5-0.8B-UD-Q2_K_XL.gguf
Qwen3.5-0.8B-UD-Q3_K_XL.gguf
Qwen3.5-0.8B-UD-Q4_K_XL.gguf
Qwen3.5-0.8B-UD-Q5_K_XL.gguf
Qwen3.5-0.8B-UD-Q6_K_XL.gguf
Qwen3.5-0.8B-UD-Q8_K_XL.gguf
"

ran=0
for name in $models; do
  model="$MODEL_DIR/$name"
  if [ ! -f "$model" ]; then
    echo "skip missing Qwen variant: $model" >&2
    continue
  fi
  inspect=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-variant-inspect.XXXXXX")
  out=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-variant.XXXXXX")
  err=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-variant-err.XXXXXX")
  trap 'rm -f "$inspect" "$out" "$err"' EXIT HUP INT TERM
  "$ROOT/ds4" --inspect -m "$model" >"$inspect"
  grep -q "qwen family: qwen35" "$inspect" || {
    echo "unexpected Qwen family for variant: $model" >&2
    exit 1
  }
  grep -q "qwen metadata prefix: qwen35" "$inspect" || {
    echo "unexpected Qwen metadata prefix for variant: $model" >&2
    exit 1
  }
  grep -q "qwen main layers: 18 recurrent, 6 full attention, 1 MTP/NextN" "$inspect" || {
    echo "unexpected Qwen layer split for variant: $model" >&2
    exit 1
  }
  grep -q "qwen hidden: embd=1024 vocab=248320 ctx=262144 main_layers=24" "$inspect" || {
    echo "unexpected Qwen hidden shape for variant: $model" >&2
    exit 1
  }
  grep -q "qwen attention: heads=8 kv_heads=2 head_dim=256 value_head_dim=256 interval=4" "$inspect" || {
    echo "unexpected Qwen attention shape for variant: $model" >&2
    exit 1
  }
  grep -q "qwen output gate type: absent" "$inspect" || {
    echo "unexpected Qwen output gate metadata for variant: $model" >&2
    exit 1
  }
  grep -q "qwen rope: dim=64 freq_base=10000000 sections=11,11,10,0" "$inspect" || {
    echo "unexpected Qwen rope shape for variant: $model" >&2
    exit 1
  }
  DS4_QWEN_COMPAT_GENERATE=0 DS4_QWEN_NATIVE_GENERATE=1 \
    "$ROOT/ds4" --metal -m "$model" -p hi -n 1 -c 1024 --temp 0 --nothink >"$out" 2>"$err"
  if grep -q "ds4: using Qwen llama.cpp compatibility backend:" "$err"; then
    echo "Qwen variant generation unexpectedly used compatibility backend: $model" >&2
    exit 1
  fi
  if ! [ -s "$out" ]; then
    echo "empty Qwen variant generation: $model" >&2
    exit 1
  fi
  rm -f "$inspect" "$out" "$err"
  trap - EXIT HUP INT TERM
  ran=$((ran + 1))
done

if [ "$ran" -eq 0 ]; then
  echo "no Qwen variants found under $MODEL_DIR" >&2
  exit 2
fi

echo "qwen35 variants smoke passed: $ran model(s)"
