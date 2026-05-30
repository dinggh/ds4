#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${QWEN35_GGUF:-/Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf}

if [ ! -x "$ROOT/ds4" ]; then
  echo "missing $ROOT/ds4; build with make ds4 first" >&2
  exit 2
fi

if [ ! -f "$MODEL" ]; then
  echo "missing Qwen GGUF: $MODEL" >&2
  exit 2
fi

out=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-bench-threshold-out.XXXXXX")
err=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-bench-threshold-err.XXXXXX")
csv=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-bench-threshold-csv.XXXXXX")
trap 'rm -f "$out" "$err" "$csv"' EXIT HUP INT TERM

QWEN35_GGUF="$MODEL" \
DS4_QWEN_COMPAT_GENERATE=1 \
QWEN_BENCH_RUNS=1 \
QWEN_BENCH_PROMPT_REPEATS=1 \
QWEN_BENCH_GEN_TOKENS=1 \
QWEN_BENCH_CSV="$csv" \
QWEN_BENCH_MIN_PREFILL_TPS=1 \
QWEN_BENCH_MIN_GEN_TPS=1 \
  "$ROOT/scripts/qwen35_bench.sh"

grep -q '^model,backend,ctx,prompt_repeats,run,gen_tokens,mtp_draft,prefill_tps,gen_tps$' "$csv"
grep -Eq '^Qwen3\.5-0\.8B-UD-Q2_K_XL\.gguf,metal,1024,1,1,1,0,[0-9.]+,[0-9.]+$' "$csv"

QWEN35_GGUF="$MODEL" \
QWEN_BENCH_RUNS=1 \
QWEN_BENCH_PROMPT_REPEATS=1 \
QWEN_BENCH_GEN_TOKENS=2 \
QWEN_BENCH_MTP_DRAFT=1 \
QWEN_BENCH_CSV="$csv" \
QWEN_BENCH_MIN_PREFILL_TPS=1 \
QWEN_BENCH_MIN_GEN_TPS=1 \
  "$ROOT/scripts/qwen35_bench.sh"

grep -q '^model,backend,ctx,prompt_repeats,run,gen_tokens,mtp_draft,prefill_tps,gen_tps$' "$csv"
grep -Eq '^Qwen3\.5-0\.8B-UD-Q2_K_XL\.gguf,metal,1024,1,1,2,1,[0-9.]+,[0-9.]+$' "$csv"

if QWEN35_GGUF="$MODEL" \
   QWEN_BENCH_RUNS=1 \
   QWEN_BENCH_PROMPT_REPEATS=1 \
   QWEN_BENCH_GEN_TOKENS=1 \
   QWEN_BENCH_MIN_GEN_TPS=not-a-number \
     "$ROOT/scripts/qwen35_bench.sh" >"$out" 2>"$err"; then
  echo "qwen benchmark invalid threshold unexpectedly passed" >&2
  exit 1
fi

grep -q 'QWEN_BENCH_MIN_GEN_TPS must be a non-negative decimal number' "$err"

if QWEN35_GGUF="$MODEL" \
   QWEN_BENCH_RUNS=1 \
   QWEN_BENCH_PROMPT_REPEATS=1 \
   QWEN_BENCH_GEN_TOKENS=1 \
   QWEN_BENCH_MIN_PREFILL_TPS=999999 \
     "$ROOT/scripts/qwen35_bench.sh" >"$out" 2>"$err"; then
  echo "qwen benchmark threshold unexpectedly passed" >&2
  exit 1
fi

grep -q 'qwen bench below threshold: prefill ' "$err"

echo "qwen35 bench threshold smoke passed: $MODEL"
