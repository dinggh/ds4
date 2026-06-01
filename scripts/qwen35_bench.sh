#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${QWEN35_GGUF:-/Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf}
BACKEND=${QWEN_BENCH_BACKEND:-metal}
CTX=${QWEN_BENCH_CTX:-1024}
GEN_TOKENS=${QWEN_BENCH_GEN_TOKENS:-16}
REPEATS=${QWEN_BENCH_PROMPT_REPEATS:-64}
RUNS=${QWEN_BENCH_RUNS:-3}
THREADS=${QWEN_BENCH_THREADS:-}
MTP_DRAFT=${QWEN_BENCH_MTP_DRAFT:-0}
MTP_APPROX_FAST_ACCEPT=${QWEN_BENCH_MTP_APPROX_FAST_ACCEPT:-0}
MTP_APPROX_DRAFT_ONLY=${QWEN_BENCH_MTP_APPROX_DRAFT_ONLY:-0}
WARM_WEIGHTS=${QWEN_BENCH_WARM_WEIGHTS:-0}
OUT=${QWEN_BENCH_CSV:-}
MIN_PREFILL_TPS=${QWEN_BENCH_MIN_PREFILL_TPS:-}
MIN_GEN_TPS=${QWEN_BENCH_MIN_GEN_TPS:-}

if [ ! -x "$ROOT/ds4" ]; then
  echo "missing $ROOT/ds4; build with make ds4 first" >&2
  exit 2
fi

if [ ! -f "$MODEL" ]; then
  echo "missing Qwen GGUF: $MODEL" >&2
  exit 2
fi

case "$BACKEND" in
  metal|cuda|cpu) ;;
  *) echo "QWEN_BENCH_BACKEND must be metal, cuda, or cpu" >&2; exit 2 ;;
esac

check_number() {
  name=$1
  value=$2
  if [ -z "$value" ]; then
    return 0
  fi
  if ! awk -v v="$value" 'BEGIN { exit(v ~ /^[0-9]+([.][0-9]+)?$/ ? 0 : 1) }'; then
    echo "$name must be a non-negative decimal number" >&2
    exit 2
  fi
}

check_min_tps() {
  name=$1
  got=$2
  want=$3
  if [ -z "$want" ]; then
    return 0
  fi
  if ! awk -v got="$got" -v want="$want" 'BEGIN { exit(got + 0 >= want + 0 ? 0 : 1) }'; then
    echo "qwen bench below threshold: $name ${got} < ${want} t/s" >&2
    exit 1
  fi
}

check_number QWEN_BENCH_MIN_PREFILL_TPS "$MIN_PREFILL_TPS"
check_number QWEN_BENCH_MIN_GEN_TPS "$MIN_GEN_TPS"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-qwen-bench.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

make_prompt() {
  count=$1
  prompt_file=$2
  {
    echo "You are evaluating Qwen native inference throughput. Answer directly."
    i=0
    while [ "$i" -lt "$count" ]; do
      echo "Benchmark paragraph $i: dense recurrent GDN layers, full attention layers, quantized matrix multiplications, and speculative NextN drafting should all remain deterministic under greedy decoding."
      i=$((i + 1))
    done
    echo "Question: summarize the throughput profile in one short sentence."
  } >"$prompt_file"
}

run_once() {
  prompt_file=$1
  run_id=$2
  log_file=$tmpdir/run-$run_id.log

  set -- "$ROOT/ds4" "--$BACKEND" -m "$MODEL" --prompt-file "$prompt_file" \
    -n "$GEN_TOKENS" -c "$CTX" --temp 0 --nothink
  if [ -n "$THREADS" ]; then
    set -- "$@" -t "$THREADS"
  fi
  if [ "$WARM_WEIGHTS" != "0" ]; then
    set -- "$@" --warm-weights
  fi
  if [ "$MTP_DRAFT" != "0" ]; then
    set -- "$@" --mtp-draft "$MTP_DRAFT"
  fi
  if [ "$MTP_APPROX_FAST_ACCEPT" != "0" ]; then
    set -- "$@" --qwen-mtp-approx-fast-accept
  fi
  if [ "$MTP_APPROX_DRAFT_ONLY" != "0" ]; then
    set -- "$@" --qwen-mtp-approx-draft-only
  fi

  if ! DS4_QWEN_COMPAT_GENERATE=0 DS4_QWEN_NATIVE_GENERATE=1 "$@" >"$log_file" 2>&1; then
    cat "$log_file" >&2
    echo "qwen bench run failed" >&2
    exit 1
  fi
  if grep -q "ds4: using Qwen llama.cpp compatibility backend:" "$log_file"; then
    cat "$log_file" >&2
    echo "qwen bench unexpectedly used compatibility backend" >&2
    exit 1
  fi

  line=$(grep "ds4: prefill:" "$log_file" | tail -n 1 || true)
  if [ -z "$line" ]; then
    cat "$log_file" >&2
    echo "qwen bench did not emit throughput line" >&2
    exit 1
  fi

  prefill=$(printf '%s\n' "$line" | sed -n 's/.*prefill: \([0-9.][0-9.]*\) t\/s, generation: \([0-9.][0-9.]*\) t\/s.*/\1/p')
  gen=$(printf '%s\n' "$line" | sed -n 's/.*prefill: \([0-9.][0-9.]*\) t\/s, generation: \([0-9.][0-9.]*\) t\/s.*/\2/p')
  if [ -z "$prefill" ] || [ -z "$gen" ]; then
    cat "$log_file" >&2
    echo "qwen bench failed to parse throughput line: $line" >&2
    exit 1
  fi
  check_min_tps prefill "$prefill" "$MIN_PREFILL_TPS"
  check_min_tps generation "$gen" "$MIN_GEN_TPS"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$model_name" "$BACKEND" "$CTX" "$repeat_count" "$run_id" "$GEN_TOKENS" "$MTP_DRAFT" "$prefill" "$gen"
}

model_name=$(basename "$MODEL")
if [ -n "$OUT" ]; then
  exec >"$OUT"
fi

printf 'model,backend,ctx,prompt_repeats,run,gen_tokens,mtp_draft,prefill_tps,gen_tps\n'

for repeat_count in $REPEATS; do
  prompt_file=$tmpdir/prompt-$repeat_count.txt
  make_prompt "$repeat_count" "$prompt_file"

  i=1
  while [ "$i" -le "$RUNS" ]; do
    run_once "$prompt_file" "$i"
    i=$((i + 1))
  done
done
