#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${QWEN35_GGUF:-/Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf}
LLAMA_CLI=${DS4_QWEN_LLAMA_CLI:-$ROOT/third_party/llama.cpp-bin/llama-b9415/llama-cli}

if [ ! -f "$MODEL" ]; then
  echo "missing Qwen GGUF: $MODEL" >&2
  exit 2
fi

if [ ! -x "$LLAMA_CLI" ]; then
  echo "missing Qwen compatibility backend; run scripts/install_qwen_compat_backend.sh" >&2
  exit 2
fi

export DS4_QWEN_LLAMA_CLI="$LLAMA_CLI"

INSPECT_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-inspect.XXXXXX")
TOKENS_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-tokens.XXXXXX")
GEN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-generate.XXXXXX")
NATIVE_GEN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-native-generate.XXXXXX")
NATIVE_GEN_ERR=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-native-generate-err.XXXXXX")
FAST_NATIVE_GEN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-fast-native-generate.XXXXXX")
FAST_NATIVE_GEN_ERR=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-fast-native-generate-err.XXXXXX")
MTP_NATIVE_GEN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-mtp-native-generate.XXXXXX")
MTP_NATIVE_GEN_ERR=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-mtp-native-generate-err.XXXXXX")
MTP_PROBE_GEN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-mtp-probe-generate.XXXXXX")
LOGITS_DUMP_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-logits.XXXXXX")
LOGPROBS_DUMP_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-logprobs.XXXXXX")
STATE_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-state.XXXXXX")
EMBED_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-embed.XXXXXX")
RMS_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-rms.XXXXXX")
QKV_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-qkv.XXXXXX")
GATE_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-attn-gate.XXXXXX")
SSM_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-ssm-out.XXXXXX")
GDN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-gdn.XXXXXX")
GDN_ZERO_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-gdn-zero.XXXXXX")
GDN_STATEFUL_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-gdn-stateful.XXXXXX")
SESSION_STEP_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-session-step.XXXXXX")
SNAPSHOT_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-snapshot.XXXXXX")
LAYER0_ZERO_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-layer0-zero.XXXXXX")
LAYER0_LOGITS_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-layer0-logits.XXXXXX")
PREFIX4_LOGITS_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-prefix4-logits.XXXXXX")
MTP_PROJ_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-mtp-proj.XXXXXX")
MTP_LAYER_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-mtp-layer.XXXXXX")
MTP_DRAFT_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-mtp-draft.XXXXXX")
FULL_ATTN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-full-attn.XXXXXX")
FFN_GATE_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-ffn-gate.XXXXXX")
FFN_DOWN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-ffn-down.XXXXXX")
FFN_OUT=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-ffn.XXXXXX")
NATIVE_ERR=$(mktemp "${TMPDIR:-/tmp}/ds4-qwen-native.XXXXXX")
trap 'rm -f "$INSPECT_OUT" "$TOKENS_OUT" "$GEN_OUT" "$NATIVE_GEN_OUT" "$NATIVE_GEN_ERR" "$FAST_NATIVE_GEN_OUT" "$FAST_NATIVE_GEN_ERR" "$MTP_NATIVE_GEN_OUT" "$MTP_NATIVE_GEN_ERR" "$MTP_PROBE_GEN_OUT" "$LOGITS_DUMP_OUT" "$LOGPROBS_DUMP_OUT" "$STATE_OUT" "$EMBED_OUT" "$RMS_OUT" "$QKV_OUT" "$GATE_OUT" "$SSM_OUT" "$GDN_OUT" "$GDN_ZERO_OUT" "$GDN_STATEFUL_OUT" "$SESSION_STEP_OUT" "$SNAPSHOT_OUT" "$LAYER0_ZERO_OUT" "$LAYER0_LOGITS_OUT" "$PREFIX4_LOGITS_OUT" "$MTP_PROJ_OUT" "$MTP_LAYER_OUT" "$MTP_DRAFT_OUT" "$FULL_ATTN_OUT" "$FFN_GATE_OUT" "$FFN_DOWN_OUT" "$FFN_OUT" "$NATIVE_ERR"' EXIT

"$ROOT/ds4" --inspect -m "$MODEL" >"$INSPECT_OUT"
grep -q "qwen family: qwen35" "$INSPECT_OUT"
grep -q "qwen metadata prefix: qwen35" "$INSPECT_OUT"
grep -q "qwen main layers: 18 recurrent, 6 full attention, 1 MTP/NextN" "$INSPECT_OUT"
grep -q "qwen hidden: embd=1024 vocab=248320 ctx=262144 main_layers=24" "$INSPECT_OUT"
grep -q "qwen attention: heads=8 kv_heads=2 head_dim=256 value_head_dim=256 interval=4" "$INSPECT_OUT"
grep -q "qwen output gate type: absent" "$INSPECT_OUT"
grep -q "qwen norm: rms_eps=9.99999997e-07" "$INSPECT_OUT"
grep -q "qwen rope: dim=64 freq_base=10000000 sections=11,11,10,0" "$INSPECT_OUT"
grep -q "qwen runtime plan: gdn_state=18.00 MiB gdn_conv=1.27 MiB full_kv_f16=3072.00 MiB at ctx=262144" "$INSPECT_OUT"
grep -q "qwen runtime per layer: recurrent_state=262144 f32 conv=18432 f32 full_kv=268435456 f16" "$INSPECT_OUT"
grep -q "qwen layer0 tensors: recurrent=true attn_qkv=q4_k attn_gate=iq3_xxs ssm_out=q5_k ffn_gate=q2_k ffn_up=q2_k ffn_down=q3_k" "$INSPECT_OUT"
grep -q "qwen layer0 gdn tensors: conv1d=f32 dt=f32 a=f32 beta=f32 alpha=f32 ssm_norm=f32" "$INSPECT_OUT"
grep -q "qwen prefix layer1 tensors: recurrent=true ffn_gate=iq3_s ffn_up=iq3_s ffn_down=iq4_xs" "$INSPECT_OUT"
grep -q "qwen prefix layer3 tensors: recurrent=false ffn_gate=iq3_s ffn_up=iq3_s ffn_down=iq4_xs" "$INSPECT_OUT"
grep -q "qwen main layer9 tensors: recurrent=true ffn_gate=iq2_s ffn_up=iq2_s ffn_down=iq3_s" "$INSPECT_OUT"
grep -q "qwen full layer7 tensors: attn_q=q2_k attn_k=q2_k attn_v=q3_k attn_output=q3_k ffn_gate=q2_k ffn_up=q2_k ffn_down=q3_k" "$INSPECT_OUT"
grep -q "qwen mtp layer24 tensors: attn_q=q2_k attn_k=q2_k attn_v=q3_k attn_output=q3_k ffn_gate=q2_k ffn_up=q2_k ffn_down=q3_k nextn_eh=q8_0 nextn_enorm=f32 nextn_hnorm=f32 shared_head_norm=f32" "$INSPECT_OUT"
grep -q "qwen first full layer: index=3 attn_q=iq3_xxs attn_k=iq3_xxs attn_v=q4_k attn_output=q3_k q_norm=f32 k_norm=f32" "$INSPECT_OUT"

"$ROOT/ds4" --dump-tokens -m "$MODEL" \
  -p '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n' >"$TOKENS_OUT"
grep -q "248045  <|im_start|>" "$TOKENS_OUT"
grep -q "248046  <|im_end|>" "$TOKENS_OUT"
grep -q "74455  assistant" "$TOKENS_OUT"

DS4_QWEN_COMPAT_GENERATE=1 "$ROOT/ds4" --metal -m "$MODEL" \
  -p 'Reply with exactly: OK' \
  -n 4 -c 1024 --temp 0 --nothink >"$GEN_OUT" 2>&1
grep -q "ds4: using Qwen llama.cpp compatibility backend:" "$GEN_OUT"
grep -q "OK" "$GEN_OUT"

"$ROOT/ds4" --metal -m "$MODEL" \
  -p 'hi' -n 1 -c 1024 --temp 0 --nothink >"$NATIVE_GEN_OUT" 2>"$NATIVE_GEN_ERR"
grep -q '^\.$' "$NATIVE_GEN_OUT"
if grep -q "ds4: using Qwen llama.cpp compatibility backend:" "$NATIVE_GEN_ERR"; then
  echo "default Qwen generation unexpectedly used compatibility backend" >&2
  exit 1
fi

DS4_QWEN_FAST_DOT=1 "$ROOT/ds4" --metal -m "$MODEL" \
  -p 'hi' -n 1 -c 1024 --temp 0 --nothink >"$FAST_NATIVE_GEN_OUT" 2>"$FAST_NATIVE_GEN_ERR"
grep -q '^\.$' "$FAST_NATIVE_GEN_OUT"
if grep -q "ds4: using Qwen llama.cpp compatibility backend:" "$FAST_NATIVE_GEN_ERR"; then
  echo "fast Qwen generation unexpectedly used compatibility backend" >&2
  exit 1
fi

"$ROOT/ds4" --metal -m "$MODEL" \
  -p 'hi' -n 2 -c 1024 --temp 0 --nothink --mtp-draft 2 >"$MTP_NATIVE_GEN_OUT" 2>"$MTP_NATIVE_GEN_ERR"
grep -q '^\.:' "$MTP_NATIVE_GEN_OUT"
if grep -q "ds4: using Qwen llama.cpp compatibility backend:" "$MTP_NATIVE_GEN_ERR"; then
  echo "MTP Qwen generation unexpectedly used compatibility backend" >&2
  exit 1
fi

DS4_MTP_PROBE=1 "$ROOT/ds4" --metal -m "$MODEL" \
  -p 'hi' -n 2 -c 1024 --temp 0 --nothink >"$MTP_PROBE_GEN_OUT" 2>&1
grep -q '^\.' "$MTP_PROBE_GEN_OUT"
grep -q '^:' "$MTP_PROBE_GEN_OUT"
grep -q "ds4: qwen mtp probe token=25" "$MTP_PROBE_GEN_OUT"

export DS4_QWEN_FAST_DOT=0

"$ROOT/ds4" --metal -m "$MODEL" -p 'hi' --dump-logits "$LOGITS_DUMP_OUT" -c 1024 >/dev/null
grep -q '"vocab":248320' "$LOGITS_DUMP_OUT"
grep -q '"argmax_token":{"id":361' "$LOGITS_DUMP_OUT"
grep -q '"argmax_logit":14.3216496' "$LOGITS_DUMP_OUT"

"$ROOT/ds4" --metal -m "$MODEL" -p 'hi' -n 1 --dump-logprobs "$LOGPROBS_DUMP_OUT" -c 1024 >/dev/null
grep -q '"prompt_tokens":19' "$LOGPROBS_DUMP_OUT"
grep -q '"top_k":20' "$LOGPROBS_DUMP_OUT"
grep -q '"step":0' "$LOGPROBS_DUMP_OUT"
grep -q '"selected":{"id":361' "$LOGPROBS_DUMP_OUT"
grep -q '"logit":14.3216496' "$LOGPROBS_DUMP_OUT"
grep -q '"logprob":-0.57126087' "$LOGPROBS_DUMP_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-state-test -c 1024 >"$STATE_OUT"
grep -q "qwen native state test: ctx=1024 gdn_state=18.00 MiB gdn_conv=1.27 MiB full_kv_f16=12.00 MiB total=31.27 MiB" "$STATE_OUT"
grep -q "qwen native state per layer: recurrent_state=262144 f32 conv=18432 f32 full_kv=1048576 f16" "$STATE_OUT"
grep -q "qwen native eval main first: token=0 checkpoint=1 layers=24 recurrent=18 full=6 logits=248320 top1=220 value=11.3058367" "$STATE_OUT"
grep -q "qwen native eval main second: token=1 checkpoint=2 layers=24 recurrent=18 full=6 logits=248320 top1=220 value=17.0067158" "$STATE_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-embed-test 0 >"$EMBED_OUT"
grep -q "qwen embed test: token=0 dim=1024 type=q6_k" "$EMBED_OUT"
grep -q "sum=-0.257391334 l2=0.723154391" "$EMBED_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-rms-test 0 >"$RMS_OUT"
grep -q "qwen rms test: token=0 dim=1024 type=f32" "$RMS_OUT"
grep -q "sum=-18.9192137 l2=39.3203736" "$RMS_OUT"
grep -q "qwen rms source: layer=0 tensor=blk.0.attn_norm.weight eps=9.99999997e-07" "$RMS_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-qkv-test 0 >"$QKV_OUT"
grep -q "qwen qkv test: token=0 dim=6144 type=q4_k" "$QKV_OUT"
grep -q "sum=-66.6622023 l2=92.2112183" "$QKV_OUT"
grep -q "qwen qkv source: layer=0 tensor=blk.0.attn_qkv.weight input=blk.0.attn_norm(token_embd) dim=6144" "$QKV_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-attn-gate-test 0 >"$GATE_OUT"
grep -q "qwen attn_gate test: token=0 dim=2048 type=iq3_xxs" "$GATE_OUT"
grep -q "sum=-273.712226 l2=39.7302985" "$GATE_OUT"
grep -q "qwen attn_gate source: layer=0 tensor=blk.0.attn_gate.weight input=blk.0.attn_norm(token_embd) dim=2048" "$GATE_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-ssm-out-test 0 >"$SSM_OUT"
grep -q "qwen ssm_out test: token=0 dim=1024 type=q5_k" "$SSM_OUT"
grep -q "sum=-4.87546518 l2=15.3201859" "$SSM_OUT"
grep -q "qwen ssm_out source: layer=0 tensor=blk.0.ssm_out.weight input=attn_qkv_value_slice offset=4096 dim=2048" "$SSM_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-gdn-param-test 0 >"$GDN_OUT"
grep -q "qwen gdn conv0 test: token=0 dim=6144 type=f32" "$GDN_OUT"
grep -q "sum=-121.28562 l2=17.0128355" "$GDN_OUT"
grep -q "qwen gdn beta test: token=0 dim=16 type=f32" "$GDN_OUT"
grep -q "sum=9.00207363 l2=2.3677989" "$GDN_OUT"
grep -q "qwen gdn g test: token=0 dim=16 type=f32" "$GDN_OUT"
grep -q "sum=-9.35391677 l2=4.95757109" "$GDN_OUT"
grep -q "qwen gdn source: layer=0 input=blk.0.attn_norm(token_embd) conv=zero_state kernel=4 dim=6144 dt_rank=16" "$GDN_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-gdn-zero-test 0 >"$GDN_ZERO_OUT"
grep -q "qwen gdn zero state test: token=0 dim=2048 type=f32" "$GDN_ZERO_OUT"
grep -q "sum=0.856073725 l2=0.275225352" "$GDN_ZERO_OUT"
grep -q "qwen gdn gated norm test: token=0 dim=2048 type=f32" "$GDN_ZERO_OUT"
grep -q "sum=-27.1083677 l2=7.68868008" "$GDN_ZERO_OUT"
grep -q "qwen gdn zero out test: token=0 dim=1024 type=q5_k" "$GDN_ZERO_OUT"
grep -q "sum=-0.761713241 l2=1.92813028" "$GDN_ZERO_OUT"
grep -q "qwen gdn zero source: layer=0 state=zero conv=silu heads=16 state_dim=128 value_head_dim=128" "$GDN_ZERO_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-gdn-stateful-test 0 1 >"$GDN_STATEFUL_OUT"
grep -q "qwen gdn stateful first out test: token=0 dim=1024 type=f32" "$GDN_STATEFUL_OUT"
grep -q "sum=-0.76171345 l2=1.92813039" "$GDN_STATEFUL_OUT"
grep -q "qwen gdn stateful second out test: token=1 dim=1024 type=f32" "$GDN_STATEFUL_OUT"
grep -q "sum=-1.12444317 l2=1.71779903" "$GDN_STATEFUL_OUT"
grep -q "qwen gdn stateful conv cache test: token=1 dim=18432 type=f32" "$GDN_STATEFUL_OUT"
grep -q "sum=-157.49872 l2=132.944665" "$GDN_STATEFUL_OUT"
grep -q "qwen gdn stateful recurrent state test: token=1 dim=262144 type=f32" "$GDN_STATEFUL_OUT"
grep -q "sum=9.40056862 l2=3.17006944" "$GDN_STATEFUL_OUT"
grep -q "qwen gdn stateful source: layer=0 tokens=0,1 conv_values=18432 state_values=262144" "$GDN_STATEFUL_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-session-step-test 0 1 -c 1024 >"$SESSION_STEP_OUT"
grep -q "qwen session step first out test: token=0 dim=1024 type=f32" "$SESSION_STEP_OUT"
grep -q "sum=-0.76171345 l2=1.92813039" "$SESSION_STEP_OUT"
grep -q "qwen session step second out test: token=1 dim=1024 type=f32" "$SESSION_STEP_OUT"
grep -q "sum=-1.12444317 l2=1.71779903" "$SESSION_STEP_OUT"
grep -q "qwen session step conv cache test: token=1 dim=18432 type=f32" "$SESSION_STEP_OUT"
grep -q "sum=-157.49872 l2=132.944665" "$SESSION_STEP_OUT"
grep -q "qwen session step recurrent state test: token=1 dim=262144 type=f32" "$SESSION_STEP_OUT"
grep -q "sum=9.40056862 l2=3.17006944" "$SESSION_STEP_OUT"
grep -q "qwen session step source: ctx=1024 layer=0 tokens=0,1 conv_values=18432 state_values=262144" "$SESSION_STEP_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-snapshot-test 0 1 -c 1024 >"$SNAPSHOT_OUT"
grep -q "qwen snapshot test: token0=0 token1=1 ctx=1024 snapshot_bytes=21211200 restored=exact replay=exact top0=220 top1=220" "$SNAPSHOT_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-layer0-zero-test 0 >"$LAYER0_ZERO_OUT"
grep -q "qwen layer0 attn residual test: token=0 dim=1024 type=f32" "$LAYER0_ZERO_OUT"
grep -q "sum=-1.01910455 l2=1.92642123" "$LAYER0_ZERO_OUT"
grep -q "qwen layer0 ffn out test: token=0 dim=1024 type=f32" "$LAYER0_ZERO_OUT"
grep -q "sum=-0.200593978 l2=0.808887697" "$LAYER0_ZERO_OUT"
grep -q "qwen layer0 zero test: token=0 dim=1024 type=f32" "$LAYER0_ZERO_OUT"
grep -q "sum=-1.21969849 l2=1.97199158" "$LAYER0_ZERO_OUT"
grep -q "qwen layer0 zero source: input=token_embd attn=zero_state_gdn residual=true ffn=dense_swiglu residual=true" "$LAYER0_ZERO_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-layer0-logits-test 0 >"$LAYER0_LOGITS_OUT"
grep -q "qwen layer0 output norm test: token=0 dim=1024 type=f32" "$LAYER0_LOGITS_OUT"
grep -q "sum=-42.4247053 l2=127.886147" "$LAYER0_LOGITS_OUT"
grep -q "qwen layer0 logits test: token=0 dim=248320 type=tied_q6_k" "$LAYER0_LOGITS_OUT"
grep -q "sum=292050.789 l2=1289.6177" "$LAYER0_LOGITS_OUT"
grep -q "qwen layer0 logits top1: token=0 value=16.5102787" "$LAYER0_LOGITS_OUT"
grep -q "qwen layer0 logits source: hidden=layer0_zero output_norm=true output=tied_q6_k vocab=248320" "$LAYER0_LOGITS_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-prefix4-logits-test 0 >"$PREFIX4_LOGITS_OUT"
grep -q "qwen prefix4 hidden test: token=0 dim=1024 type=f32" "$PREFIX4_LOGITS_OUT"
grep -q "sum=0.362500109 l2=4.4572602" "$PREFIX4_LOGITS_OUT"
grep -q "qwen prefix4 output norm test: token=0 dim=1024 type=f32" "$PREFIX4_LOGITS_OUT"
grep -q "sum=-7.00726733 l2=130.680021" "$PREFIX4_LOGITS_OUT"
grep -q "qwen prefix4 logits test: token=0 dim=248320 type=tied_q6_k" "$PREFIX4_LOGITS_OUT"
grep -q "sum=212002.575 l2=1240.87767" "$PREFIX4_LOGITS_OUT"
grep -q "qwen prefix4 logits top1: token=115057 value=13.2389956" "$PREFIX4_LOGITS_OUT"
grep -q "qwen prefix4 logits source: token=0 layers=0-3 recurrent=3 full=1 output=tied_q6_k vocab=248320" "$PREFIX4_LOGITS_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-mtp-proj-test 0 >"$MTP_PROJ_OUT"
grep -q "qwen mtp proj test: token=0 dim=1024 type=q8_0" "$MTP_PROJ_OUT"
grep -q "sum=10.7147863 l2=7.39323016" "$MTP_PROJ_OUT"
grep -q "qwen mtp proj source: layer=24 input=nextn.enorm(token_embd)+nextn.hnorm(token_embd) eh_proj=q8_0 shared_head_norm=f32 dim=1024" "$MTP_PROJ_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-mtp-layer-test 0 >"$MTP_LAYER_OUT"
grep -q "qwen mtp layer proj test: token=0 dim=1024 type=q8_0" "$MTP_LAYER_OUT"
grep -q "sum=10.7147863 l2=7.39323016" "$MTP_LAYER_OUT"
grep -q "qwen mtp layer attn test: token=0 dim=1024 type=f32" "$MTP_LAYER_OUT"
grep -q "sum=-8.95857381 l2=20.7720459" "$MTP_LAYER_OUT"
grep -q "qwen mtp layer ffn test: token=0 dim=1024 type=f32" "$MTP_LAYER_OUT"
grep -q "sum=-4.67916736 l2=9.42485426" "$MTP_LAYER_OUT"
grep -q "qwen mtp layer test: token=0 dim=1024 type=f32" "$MTP_LAYER_OUT"
grep -q "sum=-2.92295431 l2=18.7384762" "$MTP_LAYER_OUT"
grep -q "qwen mtp layer source: layer=24 input=nextn_eh_proj(token_embd,token_embd) attn=full_self ffn=dense_swiglu residual=true" "$MTP_LAYER_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-mtp-draft-test 0 >"$MTP_DRAFT_OUT"
grep -q "qwen mtp draft hidden test: token=0 dim=1024 type=f32" "$MTP_DRAFT_OUT"
grep -q "sum=-32.3766573 l2=44.2524108" "$MTP_DRAFT_OUT"
grep -q "qwen mtp draft logits test: token=0 dim=248320 type=tied_q6_k" "$MTP_DRAFT_OUT"
grep -q "sum=33178.2294 l2=1119.11986" "$MTP_DRAFT_OUT"
grep -q "qwen mtp draft top1: token=216308 value=11.6967096 main_top1=220 main_value=11.3058367 checkpoint=1" "$MTP_DRAFT_OUT"
grep -q "qwen mtp draft source: layer=24 input=last_token_embd+main_hidden head=tied_q6_k norm=f32 no_alloc=true" "$MTP_DRAFT_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-full-attn-test 0 >"$FULL_ATTN_OUT"
grep -q "qwen full attn q test: token=0 dim=2048 type=f32" "$FULL_ATTN_OUT"
grep -q "sum=-130.176962 l2=64.670139" "$FULL_ATTN_OUT"
grep -q "qwen full attn k test: token=0 dim=512 type=f32" "$FULL_ATTN_OUT"
grep -q "sum=20.9201752 l2=31.988612" "$FULL_ATTN_OUT"
grep -q "qwen full attn v test: token=0 dim=512 type=q4_k" "$FULL_ATTN_OUT"
grep -q "sum=-8.17727297 l2=12.2417466" "$FULL_ATTN_OUT"
grep -q "qwen full attn heads test: token=0 dim=2048 type=f32" "$FULL_ATTN_OUT"
grep -q "sum=-6.64066215 l2=12.4629059" "$FULL_ATTN_OUT"
grep -q "qwen full attn out test: token=0 dim=1024 type=q3_k" "$FULL_ATTN_OUT"
grep -q "sum=-0.364631704 l2=4.19457509" "$FULL_ATTN_OUT"
grep -q "qwen full attn source: layer=3 self_token=true q=iq3_xxs k=iq3_xxs v=q4_k output=q3_k heads=8 kv_heads=2 head_dim=256" "$FULL_ATTN_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-ffn-gate-test 0 >"$FFN_GATE_OUT"
grep -q "qwen ffn_gate test: token=0 dim=3584 type=q2_k" "$FFN_GATE_OUT"
grep -q "sum=1.46733758 l2=0.488408476" "$FFN_GATE_OUT"
grep -q "qwen ffn_gate source: layer=0 tensor=blk.0.ffn_gate.weight input=token_embd dim=3584" "$FFN_GATE_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-ffn-down-test 0 >"$FFN_DOWN_OUT"
grep -q "qwen ffn_down test: token=0 dim=1024 type=q3_k" "$FFN_DOWN_OUT"
grep -q "sum=-0.00955541704 l2=0.127571207" "$FFN_DOWN_OUT"
grep -q "qwen ffn_down source: layer=0 tensor=blk.0.ffn_down.weight input=blk.0.ffn_gate(token_embd) dim=3584" "$FFN_DOWN_OUT"

"$ROOT/ds4" --metal -m "$MODEL" --qwen-ffn-test 0 >"$FFN_OUT"
grep -q "qwen ffn test: token=0 dim=1024 type=f32" "$FFN_OUT"
grep -q "sum=0.930320823 l2=0.984821068" "$FFN_OUT"
grep -q "qwen ffn source: layer=0 input=blk.0.ffn_norm(token_embd) gate=q2_k up=q2_k down=q3_k intermediate=3584" "$FFN_OUT"

if "$ROOT/ds4" --metal -m "$MODEL" --qwen-moe-ffn-test 0 >"$NATIVE_ERR" 2>&1; then
  echo "Qwen MoE diagnostic unexpectedly succeeded on dense model" >&2
  exit 1
fi
grep -q "qwen-moe-ffn-test requires a Qwen MoE model" "$NATIVE_ERR"

if "$ROOT/ds4" --metal -m "$MODEL" -p hi --head-test >"$NATIVE_ERR" 2>&1; then
  echo "Qwen native-only diagnostic unexpectedly succeeded" >&2
  exit 1
fi
grep -q "Qwen head test is not supported by the native Qwen CLI path yet" "$NATIVE_ERR"

echo "qwen35 gguf smoke passed: $MODEL"
