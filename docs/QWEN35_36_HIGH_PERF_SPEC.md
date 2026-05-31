# Qwen3.5/Qwen3.6 High-Performance Inference Spec

Status: draft, spec-driven development baseline  
Date: 2026-05-30  
Owner: ds4 runtime

Implementation note: the current tree contains the native Qwen GGUF
inspector/tokenizer entry point, native Qwen generation, an explicit llama.cpp
compatibility backend for cross-checking one-shot generation, and first CUDA
single-token matvec kernels for the Qwen3.6 dense GGUF formats available on the
Spark host. The compatibility backend is intentionally isolated: it proves the
model file, tokenizer, CLI, and Qwen GGUF workflow while native Gated
DeltaNet/Metal/CUDA kernels are implemented.

This document specifies how to add high-performance inference support for
Qwen3.5 and Qwen3.6 dense and MoE text models. The goal is not generic Qwen
compatibility. The goal is a model-family-specific runtime path with the same
quality bar as the current DeepSeek V4 path: strict model contracts, specialized
GGUF layout, validated logits, backend-specific kernels, and performance-driven
state/cache design.

## 1. Scope

### 1.1 In Scope

- Qwen3.5 and Qwen3.6 text-only models using the `qwen3_5` and
  `qwen3_5_moe` architecture families.
- Dense models, for example `Qwen/Qwen3.6-27B`.
- MoE models, for example `Qwen/Qwen3.6-35B-A3B` and
  `Qwen/Qwen3.5-397B-A17B`.
- Hybrid attention execution: Gated DeltaNet linear attention layers plus full
  attention layers.
- MTP/NextN speculative decoding when the model ships MTP blocks.
- Dedicated GGUF conversion and validation for this runtime.
- CPU correctness path, Metal high-performance path, and CUDA high-performance
  path.

### 1.2 Out of Scope For First Delivery

- Vision or multimodal projectors.
- Arbitrary GGUF support.
- Arbitrary Transformers architectures named Qwen.
- Distributed serving API parity with vLLM/SGLang. Distributed expert
  parallelism is a later performance milestone, not a requirement for the first
  local runtime.

## 2. Industry Findings

Research refresh, 2026-05-30:

- Hugging Face's Qwen3.5 model documentation explicitly covers both Qwen3.5
  and Qwen3.6 dense variants and states that Qwen3.6 checkpoints share the
  Qwen3.5 architecture and `model_type`.
- Current Qwen3.6 model configs keep the top-level dense/MoE `model_type` values
  as `qwen3_5` / `qwen3_5_moe`, while the nested text configs use
  `qwen3_5_text` / `qwen3_5_moe_text`. The loader treats all four HF-style names
  plus llama.cpp's `qwen35` / `qwen35moe` names as the same runtime family.
- Hugging Face's Qwen3.5 dense reference implementation defines full-attention
  Q/K/V projections with `head_dim`, while `linear_key_head_dim` and
  `linear_value_head_dim` only belong to the Gated DeltaNet path. Therefore
  `attention.value_length != attention.key_length` is not a current public
  Qwen3.5/Qwen3.6 full-attention contract; different K/V dimensions are a
  linear-attention/GDN concern.
- Transformers' current `qwen3_5_moe` implementation keeps MoE routing in the
  FFN block: router logits are softmaxed, top-k expert weights are normalized,
  routed experts run fused gate/up then down projections, and a sigmoid-gated
  shared expert is added to the routed output.
- NVIDIA NeMo/Megatron Bridge documents Qwen3.6 as reusing the Qwen3.5 VL MoE
  architecture class and describes the family as a hybrid GDN plus standard
  attention architecture with SwiGLU, RMSNorm, top-k routing, and shared
  experts.
- Recent vLLM/llama.cpp issue traffic shows the same risk areas this spec
  gates explicitly: hybrid GDN/full-attention state correctness, FP8/KV scale
  handling, backend-specific GDN kernels, and qwen35moe prompt/KV reuse.

Primary references used by this spec:

- Hugging Face Transformers Qwen3.5 docs:
  https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Hugging Face Transformers Qwen3.5-MoE docs:
  https://huggingface.co/docs/transformers/main/model_doc/qwen3_5_moe
- Hugging Face Transformers Qwen3.5 dense source:
  https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Hugging Face Transformers Qwen3.5-MoE source:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
- vLLM Qwen3.5/Qwen3.6 serving guide:
  https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- NVIDIA NeMo/Megatron Bridge Qwen3.5 guide:
  https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/models/vlm/qwen35-vl.html

### 2.1 Architecture Is Shared Across Qwen3.5 and Qwen3.6

Official Qwen3.6 configs identify the dense architecture as
`Qwen3_5ForConditionalGeneration` with top-level `model_type = qwen3_5` and
nested text `model_type = qwen3_5_text`. The MoE architecture is
`Qwen3_5MoeForConditionalGeneration` with top-level
`model_type = qwen3_5_moe` and nested text `model_type = qwen3_5_moe_text`.
Qwen3.6 therefore reuses the Qwen3.5 architecture family instead of introducing
a separate runtime class.

The core decoder pattern is hybrid:

- Most layers are `linear_attention`.
- Every `full_attention_interval` layer is full attention; current public
  Qwen3.6 configs use interval 4.
- Linear layers use Gated DeltaNet with causal depthwise convolution and a
  recurrent state.
- Full attention layers use Q/K RMSNorm, interleaved MRoPE, and a sigmoid gate
  on the attention output.
- Dense and MoE variants share the same attention trunk. The feed-forward block
  is the main dispatch difference.

### 2.2 vLLM Lessons

vLLM implements Qwen3.5/3.6 as a hybrid model where decoder layers dispatch by
`layer_type`. Linear attention uses `QwenGatedDeltaNetAttention`, dense FFN uses
`Qwen3NextMLP`, and MoE uses `Qwen3NextSparseMoeBlock`.

Important performance lessons:

- GDN prefill is backend-sensitive. vLLM routes CUDA Hopper and some Blackwell
  configurations to FlashInfer/Cutlass paths and otherwise falls back to
  Triton/FLA.
- Decode performance depends on fused recurrent GDN update kernels.
- vLLM recommends `mamba-cache-mode=align` for this family rather than an
  unbounded all-cache mode.
- Serving recommendations for MoE include expert parallelism, prefix caching,
  and official FP8 checkpoints.
- MTP can reduce latency but lowers maximum throughput; it should be optional
  and benchmarked separately.

### 2.3 SGLang Lessons

SGLang's Qwen3.5/3.6 deployment docs reinforce the same serving constraints:

- Qwen3.6 ships both 35B-A3B MoE and 27B dense variants on the same Gated
  DeltaNet hybrid backbone.
- Qwen3.5-397B-A17B is treated as a large sparse MoE with 397B total and 17B
  active parameters.
- MTP is a first-class latency optimization, not a correctness requirement.
- The mamba/GDN cache scheduler has real memory-throughput tradeoffs:
  `no_buffer` uses less memory and is AMD-compatible, while `extra_buffer`
  enables overlap scheduling and branch-point caching on NVIDIA FLA kernels.
- Context defaults to 262144 tokens, with 128K called out as a lower practical
  bound for preserving long thinking contexts.

For this project, the actionable point is that Qwen performance tuning must
expose explicit GDN cache scheduling and speculative decoding switches instead
of hiding them behind a generic KV cache abstraction.

### 2.4 Transformers Lessons

Transformers is the correctness reference. Its Qwen3.5 and Qwen3.5-MoE
implementations expose fallback PyTorch GDN equations that are suitable for
building CPU test vectors.

The recurrent GDN update is:

```text
S_t = exp(g_t) * S_{t-1}
kv_t = sum(S_t * k_t)
delta_t = (v_t - kv_t) * beta_t
S_t = S_t + outer(k_t, delta_t)
o_t = sum(S_t * q_t)
```

where:

```text
beta_t = sigmoid(b_t)
g_t = -exp(A_log) * softplus(a_t + dt_bias)
q_t = l2norm(q_t)
k_t = l2norm(k_t)
```

The convolution input is the concatenated Q/K/V projection, passed through a
causal grouped 1D convolution and SiLU before being split into Q, K, and V.

### 2.5 llama.cpp/ggml Lessons

llama.cpp has dedicated architectures `qwen35` and `qwen35moe`, while upstream
HF-style model configs use `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, and
`qwen3_5_moe_text`. ds4 treats these naming schemes as the same
Qwen3.5/Qwen3.6 runtime family and selects the metadata prefix present in the
GGUF. llama.cpp marks these as hybrid architectures, uses
interleaved MRoPE, and has recurrent rollback support. Its conversion code is
particularly important:

- It writes GDN hyperparameters into SSM-style GGUF keys.
- It transforms `A_log` into `-exp(A_log)` at conversion time.
- It splits fused HF projections into runtime tensor names.
- It reorders Qwen3.5 V heads so the runtime can avoid expensive interleaved
  repeats.
- It appends MTP/NextN blocks after the main decoder stack and marks them
  non-recurrent.

llama.cpp's MoE implementation confirms that Qwen3.5/3.6 MoE layers combine
routed experts with a gated shared expert:

- Router tensor: per-token expert logits.
- Routed expert tensors: gate/up/down expert weights.
- Shared expert tensors: gate/up/down plus a scalar sigmoid gate.
- Routed expert weights use softmax gating.

## 3. Model Family Contract

### 3.1 Runtime Families

The runtime MUST introduce a separate Qwen family path instead of extending the
DeepSeek V4 shape tables.

```text
DS4_FAMILY_DEEPSEEK_V4
DS4_FAMILY_QWEN35_DENSE
DS4_FAMILY_QWEN35_MOE
```

Qwen3.6 models MUST be loaded through the Qwen3.5 architecture family because
their public configs use the same model types.

### 3.2 Required Metadata

The Qwen loader MUST validate these metadata fields before allocating runtime
state:

```text
qwen35.block_count
qwen35.context_length
qwen35.embedding_length
qwen35.feed_forward_length                 dense only
qwen35.expert_count                        MoE only
qwen35.expert_used_count                   MoE only
qwen35.expert_feed_forward_length          MoE only
qwen35.expert_shared_feed_forward_length   MoE only, optional if inferable
qwen35.vocab_size                             optional if token_embd infers it
qwen35.attention.head_count
qwen35.attention.head_count_kv
qwen35.attention.key_length
qwen35.attention.value_length
qwen35.attention.layer_norm_rms_epsilon
qwen35.rope.freq_base
qwen35.rope.dimension_count
qwen35.rope.dimension_sections
qwen35.full_attention_interval
qwen35.ssm.conv_kernel
qwen35.ssm.inner_size
qwen35.ssm.state_size
qwen35.ssm.time_step_rank
qwen35.ssm.group_count
qwen35.nextn_predict_layers
qwen35.output_gate_type                    optional; inspect reports absent if missing
```

The loader SHOULD also accept canonical GGUF names used by llama.cpp
(`qwen35.*`, `qwen35moe.*`, and generic `ssm.*`) if the converter emits them,
but the internal runtime structure MUST normalize them into a typed Qwen shape.

For current Qwen3.5/Qwen3.6 public full-attention layers,
`attention.value_length` MUST equal `attention.key_length`. The runtime MUST
fail loudly if they differ until a real public checkpoint and reference trace
prove a different full-attention V head dimension. Do not confuse this with GDN:
linear attention explicitly supports independent `linear_key_head_dim` and
`linear_value_head_dim`, and this implementation MUST continue to size GDN
state from those SSM metadata values.

### 3.3 Layer Classification

For `n_main = block_count - nextn_predict_layers`, layer `i` is recurrent if:

```text
i < n_main && ((i + 1) % full_attention_interval != 0)
```

MTP/NextN layers MUST be non-recurrent full-attention decoder blocks and MUST
not execute during the main forward pass.

### 3.4 Known Public Shapes

These are hard constraints for shape validation, not performance assumptions:

| Model | Family | Layers | Hidden | Context | Linear Pattern | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3.6-27B | dense | 64 | 5120 | 262144 | 3 GDN + 1 full | 24 Q heads, 4 KV heads, MTP=1 |
| Qwen3.6-35B-A3B | MoE | 40 | 2048 | 262144 | 3 GDN + 1 full | 256 experts, top-8 routed, shared expert, MTP=1 |
| Qwen3.5-397B-A17B | MoE | 60 | 4096 | 262144 | 3 GDN + 1 full | 512 experts, top-10 routed, MTP=1 |

The implementation MUST not rely only on this table. It MUST validate from
metadata and tensor shapes.

## 4. Tensor Contract

The project SHOULD use a Qwen-specific GGUF layout rather than accepting raw HF
names at runtime. The converter owns all projection splitting, transposition,
V-head reorder, and scalar transforms.

### 4.1 Common Tensors

```text
token_embd.weight          [n_embd, n_vocab]
output_norm.weight         [n_embd]
output.weight              [n_embd, n_vocab] optional; tie to token_embd if absent
blk.N.attn_norm.weight     [n_embd]
blk.N.attn_post_norm.weight[n_embd]
```

### 4.2 Full Attention Layer Tensors

Full attention layers MUST include:

```text
blk.N.attn_q.weight        [n_embd, 2 * n_head * head_dim]
blk.N.attn_k.weight        [n_embd, n_kv_head * head_dim]
blk.N.attn_v.weight        [n_embd, n_kv_head * head_dim]
blk.N.attn_output.weight   [n_head * head_dim, n_embd]
blk.N.attn_q_norm.weight   [head_dim]
blk.N.attn_k_norm.weight   [head_dim]
```

The `attention.value_length` metadata is validated, but for the current public
architecture it is expected to match `head_dim`. The full-attention KV cache is
therefore two f16 planes of `[n_kv_head * head_dim]` per token. Any future
checkpoint that separates full-attention K/V dimensions requires a coordinated
contract change for `attn_v.weight`, `attn_output.weight`, the sigmoid gate
shape, KV cache serialization, MTP, and backend kernels before it can be
accepted.

`attn_q.weight` contains query and gate halves. The runtime MUST split it into:

```text
Q    [head_dim, n_head, n_tokens]
gate [head_dim, n_head, n_tokens]
```

The full attention output MUST be multiplied by `sigmoid(gate)` before the
output projection.

### 4.3 Linear Attention / Gated DeltaNet Tensors

Recurrent layers MUST include:

```text
blk.N.attn_qkv.weight      [n_embd, 2 * key_dim + value_dim]
blk.N.attn_gate.weight     [n_embd, value_dim]
blk.N.ssm_conv1d.weight    [conv_kernel, 2 * key_dim + value_dim]
blk.N.ssm_dt.bias          [num_v_heads]
blk.N.ssm_a                [num_v_heads]      converted to -exp(A_log)
blk.N.ssm_beta.weight      [n_embd, num_v_heads]
blk.N.ssm_alpha.weight     [n_embd, num_v_heads]
blk.N.ssm_norm.weight      [head_v_dim]
blk.N.ssm_out.weight       [value_dim, n_embd]
```

Definitions:

```text
key_dim      = linear_num_key_heads   * linear_key_head_dim
value_dim    = linear_num_value_heads * linear_value_head_dim
conv_dim     = 2 * key_dim + value_dim
head_k_dim   = linear_key_head_dim
head_v_dim   = linear_value_head_dim
num_k_heads  = linear_num_key_heads
num_v_heads  = linear_num_value_heads
```

The converter MUST store `ssm_a = -exp(A_log)` so runtime GDN can compute:

```text
g = softplus(alpha + dt_bias) * ssm_a
```

### 4.4 Dense FFN Tensors

Dense models MUST include:

```text
blk.N.ffn_gate.weight      [n_embd, n_ff]
blk.N.ffn_up.weight        [n_embd, n_ff]
blk.N.ffn_down.weight      [n_ff, n_embd]
```

The activation is SwiGLU/SiLU:

```text
ffn(x) = down(silu(gate(x)) * up(x))
```

### 4.5 MoE FFN Tensors

MoE models MUST include routed experts:

```text
blk.N.ffn_gate_inp.weight  [n_embd, n_expert]
blk.N.ffn_gate_exps.weight [n_embd, n_ff_exp, n_expert]
blk.N.ffn_up_exps.weight   [n_embd, n_ff_exp, n_expert]
blk.N.ffn_down_exps.weight [n_ff_exp, n_embd, n_expert]
```

Converters MAY emit fused routed gate/up experts:

```text
blk.N.ffn_gate_up_exps.weight [n_embd, 2 * n_ff_exp, n_expert]
```

MoE models SHOULD include a shared expert when present in the checkpoint:

```text
blk.N.ffn_gate_inp_shexp.weight [n_embd]
blk.N.ffn_gate_shexp.weight     [n_embd, n_ff_shexp]
blk.N.ffn_up_shexp.weight       [n_embd, n_ff_shexp]
blk.N.ffn_down_shexp.weight     [n_ff_shexp, n_embd]
```

Execution:

```text
routed = softmax(router(x)).topk(expert_used_count)
shared = shared_down(silu(shared_gate(x)) * shared_up(x))
shared = sigmoid(shared_scalar_gate(x)) * shared
out = routed_experts(x, routed) + shared
```

### 4.6 MTP/NextN Tensors

If `nextn_predict_layers > 0`, MTP tensors MUST be appended after the main
decoder layers and MUST include:

```text
blk.M.nextn_eh_proj.weight         [2 * n_embd, n_embd]
blk.M.nextn_enorm.weight           [n_embd]
blk.M.nextn_hnorm.weight           [n_embd]
blk.M.nextn_embed_tokens.weight    [n_embd, n_vocab] optional
blk.M.nextn_shared_head_head.weight[n_embd, n_vocab] optional
blk.M.nextn_shared_head_norm.weight[n_embd] optional
```

The MTP block uses full attention and the same dense/MoE FFN type as the model.

## 5. Execution Graph Spec

### 5.1 Main Decoder

For each main layer:

```text
x0 = x
h  = rms_norm(x, attn_norm)

if recurrent_layer:
    a = qwen_gdn(h, layer, recurrent_state)
else:
    a = qwen_full_attention(h, layer, kv_cache)

x = x0 + a
y0 = x
h  = rms_norm(x, attn_post_norm)

if dense:
    f = dense_ffn(h)
else:
    f = qwen_moe(h)

x = y0 + f
```

Final:

```text
h = rms_norm(x, output_norm)
logits = output(h)
```

### 5.2 Full Attention

The full attention path MUST:

1. Project query+gate with the joint Q projection.
2. Split Q and gate.
3. Apply RMSNorm to Q and K per head.
4. Apply interleaved MRoPE using `rope.dimension_sections`.
5. Use standard causal attention over full-attention-layer KV cache.
6. Multiply attention output by `sigmoid(gate)`.
7. Apply output projection.

### 5.3 Gated DeltaNet Linear Attention

The GDN path MUST:

1. Project QKV into `qkv_mixed`.
2. Project `z` from `attn_gate.weight`.
3. Project `beta = sigmoid(ssm_beta(x))`.
4. Project `alpha = ssm_alpha(x)`.
5. Compute `g = softplus(alpha + dt_bias) * ssm_a`.
6. Run causal grouped 1D convolution over `qkv_mixed`.
7. Apply SiLU to convolution output.
8. Split into Q, K, V.
9. Apply L2 normalization to Q and K.
10. Run recurrent GDN update.
11. Apply gated RMSNorm: `rms_norm(output, ssm_norm) * silu(z)`.
12. Apply `ssm_out.weight`.

The implementation MUST preserve exact recurrent state semantics across
prefill, decode, rollback, and speculative decoding.

## 6. State And Cache Spec

The Qwen runtime MUST maintain separate state stores:

```text
full attention KV cache: only for full-attention layers
GDN conv state:         only for recurrent layers
GDN recurrent state:    only for recurrent layers
MTP KV cache:           only for MTP draft graph
```

State shapes:

```text
conv_state[layer][seq]      = [conv_dim, conv_kernel - 1 + spec_slots]
gdn_state[layer][seq]       = [num_v_heads, head_v_dim, head_k_dim]
kv_cache[layer][seq]        = [tokens, n_kv_heads, head_dim] for K and V
```

The implementation SHOULD support recurrent rollback snapshots for
speculative decoding. Prefix-cache support SHOULD use an aligned mode similar
to vLLM's `mamba-cache-mode=align`.

## 7. High-Performance Implementation Plan

### 7.1 CPU Correctness Path

The CPU path MUST be simple and exact before GPU optimization:

- Implement Qwen tokenizer/chat-template handling or validate compatibility
  with the existing tokenizer layer.
- Implement strict GGUF metadata and tensor validation.
- Implement dense and MoE CPU forward for single-token decode and short
  prefill.
- Generate test vectors from Transformers for GDN micro-kernels and full logits.

### 7.2 Metal Path

Metal is the first local high-performance target.

Required kernels:

- Full-attention decode over only full-attention layers.
- Qwen GDN single-token decode fused kernel:
  - QKV/z/beta/alpha matmul outputs can remain separate at first.
  - Conv state update, causal convolution, SiLU, Q/K L2Norm, GDN recurrent
    update, gated RMSNorm, and state writeback SHOULD be fused in the optimized
    version.
- Qwen GDN chunk prefill kernel:
  - chunked recurrent scan for long prompt ingestion.
  - explicit state output for the next decode token.
- MoE router/top-k kernel.
- Expert matmul batching:
  - group tokens by expert for prefill.
  - direct top-k expert execution for decode.
  - keep router, norms, GDN, and full attention at F16/BF16 or Q8-quality.

Avoid building this as a generic graph interpreter. The hot path SHOULD use
fixed, validated Qwen shapes and backend-specific tensor pointer tables, as the
DeepSeek V4 path does today.

### 7.3 CUDA Path

CUDA SHOULD follow the same runtime contract with CUDA-specific kernels:

- fused GDN decode, modeled after ggml's fused gated-delta-net kernel and vLLM's
  recurrent update paths.
- chunked GDN prefill for throughput.
- FP8/BF16 input paths where the official checkpoint supports it.
- expert-parallel hooks for multi-GPU MoE serving.

### 7.4 MoE Performance Requirements

MoE must not be implemented as a naive per-token, per-expert loop.

Required:

- router logits in one batched matmul.
- top-k selection with stable deterministic tie behavior.
- expert token bucketing for prefill.
- decode path optimized for top-k active experts.
- shared expert computed once per token and gated by sigmoid.
- quantized expert tensors loaded in a layout that minimizes dequant scatter.

Recommended:

- expert hotness counters.
- optional expert paging/offload for very large Qwen3.5 MoE checkpoints.
- per-layer expert residency policy for memory-constrained machines.

## 8. Quantization And Conversion

The converter is part of the runtime contract.

### 8.1 Converter Must

- Read HF config and tokenizer metadata.
- Emit normalized Qwen metadata.
- Split fused HF projections into runtime tensors.
- Convert `A_log` to `-exp(A_log)`.
- Squeeze 1D convolution weights to `[kernel, channels]`.
- Reorder V heads for Qwen3.5/Qwen3.6 GDN so runtime broadcast is contiguous.
- Append MTP/NextN blocks after main decoder blocks.
- Preserve tokenizer special tokens and Qwen chat-template behavior.

### 8.2 Quantization Policy

Recommended first policies:

- Attention, GDN state projections, router, norms, and output head: Q8/F16
  quality first.
- Dense FFN: Q4/Q5 after correctness.
- Routed MoE experts: Q4-family first, then IQ/imatrix experiments.
- Shared expert: Q8 or higher-quality Q4 because it runs for every token.
- GDN recurrence state: F16/BF16, not quantized.
- Conv state: F16/BF16.

Each quantization recipe MUST have a logits acceptance threshold and benchmark
result before becoming default.

## 9. Validation Spec

### 9.1 Correctness Tests

The implementation MUST include:

- tokenizer and chat-template golden tests:
  - `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, tool-call tokens.
- GDN micro-tests:
  - conv update.
  - recurrent state update.
  - chunk prefill vs token-by-token decode equivalence.
- full-attention tests:
  - interleaved MRoPE sections.
  - Q/K RMSNorm.
  - sigmoid attention gate.
- dense FFN logits tests.
- MoE router/top-k tests:
  - selected expert ids.
  - expert weights.
  - shared expert gate.
- full model logits tests:
  - short prompt.
  - multi-turn chat prompt.
  - long-context prompt crossing several full-attention and recurrent layers.
- MTP tests:
  - draft logits.
  - rollback state restoration.

### 9.2 Performance Benchmarks

Benchmarks MUST report:

- prompt length: 128, 2k, 8k, 32k, 128k when model context allows.
- decode throughput: tok/s at batch 1.
- prefill throughput: tok/s.
- time to first token.
- memory footprint by tensor class:
  - weights.
  - full attention KV.
  - GDN conv state.
  - GDN recurrent state.
  - MoE expert residency/offload.
- backend:
  - CPU.
  - Metal.
  - CUDA.

## 10. Milestones

### M0: Spec And Research

Acceptance:

- This document exists in repo.
- Sources and architecture findings are documented.
- Loader/runtime boundaries are explicit.

### M1: Qwen GGUF Inspector And Converter Skeleton

Acceptance:

- Converter reads Qwen3.5/Qwen3.6 dense and MoE HF configs.
- Converter emits normalized Qwen metadata.
- Inspector validates metadata, layer classification, and tensor presence.
- No inference required.

Current implementation:

- `./ds4 --inspect -m /Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf`
  validates the real Unsloth Qwen3.5-0.8B MTP GGUF.
- The inspected file reports family `qwen35`, metadata prefix `qwen35`, 24 main
  layers, 18 recurrent GDN layers, 6 full-attention layers, and 1 MTP/NextN
  layer.
- The loader accepts both llama.cpp-style `qwen35`/`qwen35moe` architecture
  names and HF-style `qwen3_5`/`qwen3_5_text`/`qwen3_5_moe`/
  `qwen3_5_moe_text` aliases, then selects whichever metadata prefix is actually
  present (`qwen35`, `qwen35moe`, `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, or
  `qwen3_5_moe_text`). This alias matrix is pinned by
  `./ds4_test --qwen-arch-aliases` and included in `make qwen-check`.
- For Gated DeltaNet metadata, the loader also falls back from
  `<qwen-prefix>.ssm.*` to generic `ssm.*` keys when a converter emits the
  llama.cpp-style SSM namespace.
- `output_gate_type` is parsed and exposed in `--inspect` when present. The
  target Unsloth Qwen3.5-0.8B GGUF does not contain this key, so the smoke test
  pins the current behavior as `qwen output gate type: absent`.
- This GGUF omits `qwen35.vocab_size`; the loader infers it from
  `token_embd.weight`.
- The loader validates tensor dimensions for recurrent GDN layers, full
  attention layers, dense FFN tensors, MoE routed experts, MoE shared experts,
  and the MTP/NextN block before the model can be used by either the
  compatibility backend or future native Qwen execution. For MTP/NextN layers it
  now binds the block's full-attention tensors, dense/MoE FFN tensors, required
  `nextn.eh_proj/enorm/hnorm` tensors, and optional shared-head tensors.
- The Qwen inspector prints dense FFN tensor contracts without changing the
  target GGUF smoke strings, and switches to explicit MoE fields (`ffn_router`,
  fused/separate routed gate-up tensors, routed down tensor, and shared expert
  presence) for `qwen35moe`/`qwen3_5_moe` models.
- MoE shared expert width is read from
  `qwen35moe.expert_shared_feed_forward_length` when present and otherwise
  inferred from `blk.0.ffn_gate_shexp.weight`.
- Qwen layer binding uses a Qwen-specific compiled layer limit rather than the
  DeepSeek V4 61-layer profile limit, so larger Qwen3.5/Qwen3.6 MoE metadata
  fails explicitly only if it exceeds that Qwen limit.
- Inspector computes the native runtime state plan from GGUF metadata:
  recurrent GDN state, causal conv state, and full-attention f16 KV cache at
  the model context length. This is the buffer contract for the later
  Metal/CUDA Qwen runtime.
- Inspector validates the Qwen full-attention positional metadata needed for
  native multi-token KV attention: `attention.value_length`,
  `rope.freq_base`, `rope.dimension_count`, and
  `rope.dimension_sections`. The target smoke GGUF reports
  `head_dim=256`, `value_head_dim=256`, `rope_dim=64`,
  `rope.freq_base=10000000`, and sections `11,11,10,0`; the section sum is
  doubled for the interleaved MRoPE dimension.
- `--qwen-state-test -c N` allocates and releases the native Qwen runtime
  buffers for a requested context size, currently covering recurrent GDN state,
  causal conv state, and full-attention f16 KV cache. This is a smoke test for
  the native session memory layout before kernels are enabled.
- Qwen now has a native session skeleton: `ds4_session_create()` allocates the
  Qwen state plan and `ds4_session_free()` releases it. `ds4_session_eval()`
  now executes the CPU reference path across all 24 main layers, updates
  recurrent GDN state, writes full-attention K/V into the native f16 cache, and
  emits Qwen logits from the session buffer.
- The state smoke asserts that session eval updates the native Qwen checkpoint
  for two tokens and produces full-main-layer logits top-1 values, so the
  native Qwen path no longer falls through into DeepSeek graph code.
- Qwen `--dump-logits`, `--dump-logprobs`, and ordinary sampled `-p/-n`
  generation now use the native session path by default instead of the llama.cpp
  compatibility backend. Setting `DS4_QWEN_COMPAT_GENERATE=1` or
  `DS4_QWEN_NATIVE_GENERATE=0` explicitly routes one-shot generation through the
  compatibility backend for cross-checks.
- Native Qwen CPU reference matvecs now use ds4's persistent row-parallel worker
  pool for Q2_K/Q3_K/Q4_K/Q5_K, IQ3_XXS, IQ2_S, IQ3_S, and IQ4_XS tensor
  formats. This is still a reference implementation, but it removes the
  single-threaded row loop from the main Qwen decode path and gives a clearer
  bridge to future SIMD/Metal/CUDA kernels.
- IQ2_S, IQ3_S, and IQ4_XS dequantization are now implemented locally in ds4
  using the ggml lookup grids/nonlinear tables and block layouts. The Qwen
  native CPU reference path no longer depends on dynamically loading ggml-base
  for row dequantization.
- The matvec hot path also has fused dot kernels for IQ2_S, IQ3_S, IQ4_XS,
  Q2_K, Q4_K, and Q5_K, so dense FFN rows plus recurrent GDN `attn_qkv`/`ssm_out`
  rows in those formats are accumulated directly from quantized blocks without
  materializing a temporary f32 row. The standalone dequant functions remain
  available for diagnostics and checksum tests. The Q2_K/Q4_K/Q5_K fused dots
  keep the same per-block value order as the earlier dequant-then-dot path,
  preserving the smoke-test logits/state checksums while removing stack dequant
  traffic from the default path.
- Q3_K/IQ3_XXS matvecs now use block-local stack dequant plus
  sequential accumulation instead of allocating a full f32 row per output row.
  This keeps the same accumulation order as the earlier reference path while
  reducing heap traffic across the GDN and attention projection hot paths.
- A scalar Q3_K exact fused-dot variant was tested and kept out of the default
  path because it preserved checksums but regressed the target GGUF fast-dot
  decode rate. Q3_K should move next through vector/SIMD or backend kernels
  rather than a scalar fused loop.
- A scalar IQ3_XXS exact fused-dot variant was also tested and kept out of the
  default path because it did not improve the target GGUF default decode rate.
  IQ3_XXS should follow the same vector/SIMD or backend-kernel route.
- The native session owns reusable `n_embd` scratch buffers for the main
  single-token decode loop (`cur`, `next`, attention output, post-attention,
  FFN output, and output norm), removing repeated top-level malloc/free pairs
  from every Qwen token evaluation.
- Recurrent GDN and dense FFN layer helpers can now use native-session scratch
  for their internal projections and activations (`qkv`, gate, up, mid, conv,
  beta/alpha/g, GDN, and gated vectors). Standalone diagnostics still allocate
  local buffers, while normal session decode avoids those per-layer malloc/free
  pairs.
- Full-attention session decode also reuses native-session scratch for `qg`,
  K/V projection vectors, per-head outputs, and causal softmax scores. The
  independent full-attention diagnostics keep their local allocations, but the
  normal decode path no longer allocates those buffers for every full-attention
  layer.
- `ds4_session_eval()` wraps native Qwen token evaluation in the same allocation
  guard used by the DeepSeek CPU decode path. Qwen sessions preallocate the
  checkpoint token buffer to `ctx_size`, so prompt prefill and decode cannot
  grow it inside the guarded hot path. The target GGUF smoke therefore fails if
  normal Qwen token eval reintroduces heap allocation.
- Native Qwen sessions also own reusable Q8_0 activation quantization scratch
  sized for the largest relevant Qwen projection input (`n_embd`,
  `2*n_embd` for NextN `eh_proj`, `n_ff_exp`, or shared-expert width). Q8_0
  output heads, MTP projection, MoE router/shared projections, 1-D shared gates,
  and routed expert slices can use this scratch under the allocation guard
  instead of allocating `xq/xscale` per token.
- Dense Qwen CPU decode no longer assumes the UD-Q2_K_XL tensor recipe for the
  common trunk. Token embeddings, recurrent GDN projections, full-attention
  projections, dense FFN projections, and tied logits can now use Q8_0, Q6_K,
  BF16, legacy Q4_0/Q4_1, K-quants, and supported IQ row formats through the
  shared dense/quant matvec helpers. `make qwen-variants-smoke` exercises the
  local Qwen3.5-0.8B dense MTP GGUF matrix when present: BF16, Q3_K/Q4_K/Q5_K,
  Q6_K, Q8_0, Q4_0/Q4_1, IQ4_NL/IQ4_XS, UD IQ2/IQ3, and UD Q2_K through Q8_K
  XL variants. The variants smoke first asserts each present GGUF reports the
  expected Qwen family, metadata prefix, 18 recurrent/6 full-attention/1 MTP
  layer split, hidden/attention/rope dimensions, and absent output-gate metadata
  before running the native generation path. Variant generation also forces
  `DS4_QWEN_COMPAT_GENERATE=0` and `DS4_QWEN_NATIVE_GENERATE=1`, then fails if
  a compatibility-backend log appears.
- `scripts/qwen35_bench.sh` and `make qwen-bench` provide a repeatable native
  Qwen throughput gate over the real CLI generation path. The script emits CSV
  rows with `prefill_tps` and `gen_tps`, supports `QWEN_BENCH_PROMPT_REPEATS`,
  `QWEN_BENCH_RUNS`, `QWEN_BENCH_GEN_TOKENS`, `QWEN_BENCH_BACKEND`,
  `QWEN_BENCH_MTP_DRAFT`, and `QWEN_BENCH_CSV`. Performance regression gates
  can be enabled by setting `QWEN_BENCH_MIN_PREFILL_TPS` and/or
  `QWEN_BENCH_MIN_GEN_TPS`; each benchmark run fails if its measured
  throughput falls below the configured threshold. `make qwen-bench-threshold-smoke`
  covers CSV emission, MTP benchmark routing via `QWEN_BENCH_MTP_DRAFT`,
  invalid threshold validation, and both the passing and failing threshold
  paths on the target GGUF. The benchmark script forces the native Qwen CLI path
  for every measured run (`DS4_QWEN_COMPAT_GENERATE=0`,
  `DS4_QWEN_NATIVE_GENERATE=1`) and fails if a compatibility-backend log appears,
  so ambient compat-backend environment variables cannot pollute throughput
  gates.
- Qwen sessions now have a Qwen-specific snapshot payload (`DSVQ`) that
  serializes checkpoint tokens, logits, recurrent GDN state, convolution cache,
  live full-attention KV rows, MTP hidden state, and optional MTP draft logits.
  `make qwen-bench-smoke` runs `ds4-bench` on the target GGUF across two
  frontiers, forcing snapshot save, decode, restore, and continued prefix
  extension through the native Qwen state. `--qwen-snapshot-test TOKEN0 TOKEN1`
  additionally byte-checks exact restore of logits/GDN/full-KV/MTP-hidden state
  and then replays `TOKEN1` to prove identical post-restore logits.
- `make qwen-check` is the top-level local Qwen gate. It runs the architecture
  alias self-test, target GGUF smoke, local dense variant smoke, benchmark
  threshold smoke, and benchmark snapshot smoke so spec-driven Qwen changes have
  one command that covers the target model, the available local tensor-format
  matrix, and the benchmark regression gate mechanics.
- The full-attention value path keeps the original sigmoid and multiply order
  for checksum stability, but hoists fixed per-head gate addressing out of the
  innermost value loop and avoids rebuilding a runtime-plan check in every GDN
  layer.
- The tied Q6_K logits head for GGUFs without `output.weight` also uses the
  row-parallel worker pool over the full vocabulary. It now uses an original
  row-order fused Q6_K dot kernel instead of materializing a full embedding row
  per vocabulary worker iteration, while preserving logits checksums.
- The Qwen native generation default enables the fast-dot path. This uses a
  faster Q6_K logits-head dot path that accumulates in a lower-overhead order,
  and lets Q2_K matvecs quantize the activation once to Q8_K and reuse ds4's
  existing Q2_K x Q8_K NEON dot kernel. For Qwen target dimensions up to 64
  QK_K blocks the Q8_K activation scratch stays on the stack, avoiding heap
  traffic in fast-dot matvecs. The Q2_K fast path is row-parallel over ds4's
  persistent worker pool, so the optimized path keeps the same scheduling shape
  as the reference matvecs instead of falling back to a single-threaded side
  path. Dense FFN gate/up pairs that are both Q2_K share one Q8_K activation
  quantization and compute both projections together through a row-parallel pair
  worker. Setting `DS4_QWEN_FAST_DOT=0` disables this path for exact checksum
  diagnostics; the smoke test covers both default fast native generation and
  exact diagnostic checksums on the target GGUF.
- An experimental `DS4_QWEN_FAST_Q4=1` hook exists for Q4_K x Q8_K matvecs, but
  it is intentionally separate from `DS4_QWEN_FAST_DOT` because the target GGUF
  is more sensitive to that approximation. It is not part of the smoke gate yet.
- An experimental `DS4_QWEN_FAST_Q3=1` hook exists for Q3_K x Q8_K matvecs under
  the same policy: useful for kernel work, but kept out of `DS4_QWEN_FAST_DOT`
  because it changes the target GGUF's greedy output.
- An experimental `DS4_QWEN_FAST_Q5=1` hook exists for Q5_K x Q8_K matvecs.
  This targets recurrent GDN `ssm_out` tensors in the current Qwen3.5
  UD-Q2_K_XL smoke model. It uses scale/min arithmetic with Q8_K block sums and
  row-parallel scheduling, but it is kept out of `DS4_QWEN_FAST_DOT` because it
  changes the target GGUF's greedy output from the checksum-stable path.
- `--qwen-embed-test TOKEN` decodes one token embedding row through the native
  CPU reference path. It currently supports F32, F16, and Q6_K embeddings; the
  target Qwen3.5-0.8B GGUF uses Q6_K for `token_embd.weight`.
- `--qwen-rms-test TOKEN` applies `blk.0.attn_norm.weight` to that native
  embedding using the model's `attention.layer_norm_rms_epsilon`, producing a
  stable checksum for the first Qwen CPU reference stage.
- Inspector prints layer-0 recurrent tensor quantization types, so projection
  reference work can be gated on the actual quant formats present in the target
  GGUF (`attn_qkv=q4_k`, `attn_gate=iq3_xxs`, `ssm_out=q5_k`,
  `ffn_gate=q2_k`, `ffn_up=q2_k`, `ffn_down=q3_k` for the current smoke
  model).
- Inspector also prints the first four prefix layers' dense FFN quantization
  types. This matters for UD-Q2_K_XL: layer 0 uses Q2_K/Q3_K, layers 1-3 use
  IQ3_S gate/up and IQ4_XS down tensors, and later layers also introduce
  IQ2_S gate/up plus IQ3_S down tensors.
- `--qwen-qkv-test TOKEN` runs the first recurrent layer's `blk.0.attn_qkv`
  projection from the native RMSNorm output. The CPU reference currently
  includes Q4_K row dequantization/matvec, which covers the target GGUF's
  `attn_qkv.weight`.
- `--qwen-attn-gate-test TOKEN` runs the first recurrent layer's
  `blk.0.attn_gate.weight` projection from the same RMSNorm output. The CPU
  reference includes IQ3_XXS row dequantization/matvec based on ggml's block
  layout and lookup grid, covering the target GGUF's recurrent gate projection.
- `--qwen-ssm-out-test TOKEN` runs `blk.0.ssm_out.weight` using Q5_K
  dequantization/matvec. Until GDN update is implemented, the diagnostic feeds
  the value slice of `attn_qkv` as a synthetic input to validate the Q5_K output
  projection format.
- `--qwen-gdn-param-test TOKEN` runs the first recurrent layer's f32 GDN
  parameter path from the native attention RMSNorm output: Q4_K qkv projection,
  zero-state causal conv checksum, f32 beta/alpha projections, sigmoid beta,
  and `g = softplus(alpha + dt_bias) * ssm_a`.
- `--qwen-gdn-zero-test TOKEN` runs a layer-0 zero-state recurrent GDN
  semantic micro-path: SiLU causal conv, Q/K L2 normalization, zero-state GDN
  update, gated per-value-head RMSNorm with `ssm_norm`, and Q5_K `ssm_out`.
  With zero previous recurrent state, the update is deterministic and avoids
  long-sequence cache semantics while still exercising the core recurrent
  equation.
- `--qwen-gdn-stateful-test TOKEN0 TOKEN1` runs two layer-0 recurrent GDN steps
  with mutable causal-conv cache and recurrent state. It applies `exp(g)` state
  decay, beta-scaled delta updates, and emits checksums for both token outputs,
  the conv cache, and the recurrent state tensor.
- `--qwen-session-step-test TOKEN0 TOKEN1` runs the same two layer-0 recurrent
  GDN steps against buffers allocated by `ds4_session_create()`, validating that
  native session `gdn_conv` and `gdn_state` layout match the standalone
  stateful reference path.
- `--qwen-full-attn-test TOKEN` runs the first full-attention layer's
  self-token native path. On the target smoke GGUF this is layer 3 and covers
  IQ3_XXS query/gate, IQ3_XXS key, Q4_K value, per-head Q/K RMSNorm,
  single-token causal attention, sigmoid gate application, and Q3_K output
  projection.
- `--qwen-ffn-gate-test TOKEN` and `--qwen-ffn-down-test TOKEN` cover dense FFN
  Q2_K and Q3_K row dequantization/matvec. The current tests use the native
  token embedding and the Q2_K gate output as deterministic synthetic inputs,
  because the full layer residual path is not implemented yet.
- `--qwen-ffn-test TOKEN` runs the native dense FFN semantic chain for layer 0:
  `post_attention_norm(token_embd)`, Q2_K gate/up projections, SwiGLU, then
  Q3_K down projection. This is still a micro-path because it uses token
  embedding as the residual input instead of the real attention output.
- `--qwen-moe-ffn-test TOKEN` runs the native MoE FFN semantic chain for layer 0
  on Qwen3.5/Qwen3.6 MoE models: `post_attention_norm(token_embd)`, router
  projection, softmax/top-k routing, routed expert SwiGLU/down projection, and
  optional shared expert contribution. Dense Qwen models must reject this
  diagnostic, and the target dense smoke GGUF asserts that rejection.
- `--qwen-layer0-zero-test TOKEN` stitches the layer-0 zero-state recurrent
  GDN output into the residual stream, then runs dense FFN on the post-attention
  residual and applies the second residual add. This is the first native Qwen
  diagnostic that exercises the recurrent block's attention and FFN halves in
  their decoder order.
- `--qwen-layer0-logits-test TOKEN` runs the layer-0 zero-state residual output
  through `output_norm.weight` and the logits head. The target smoke GGUF uses a
  tied embedding head, so this diagnostic supports `token_embd.weight` Q6_K as
  the output matrix when `output.weight` is absent.
- `--qwen-prefix4-logits-test TOKEN` runs the ordered native prefix for the
  target Qwen3.5 dense smoke model: recurrent stateful GDN + dense FFN for
  layers 0-2, full-attention self-token + dense FFN for layer 3, final
  `output_norm.weight`, and the tied Q6_K logits head. This is the first mixed
  recurrent/full-attention Qwen execution checkpoint. The CPU reference now
  accepts IQ2_S/IQ3_S/IQ4_XS dense FFN tensors for layers after layer 0; the
  current transitional implementation calls ggml's dequant functions for those
  formats while native SIMD/GPU kernels are still pending.
- `--qwen-mtp-proj-test TOKEN` runs the first native MTP/NextN micro-path on the
  appended MTP block: `nextn.enorm(token_embd)`, `nextn.hnorm(token_embd)`,
  concatenation, and `nextn.eh_proj`. On the target GGUF this exercises the
  layer-24 Q8_0 `nextn.eh_proj.weight` and pins its checksum in smoke.
- `--qwen-mtp-layer-test TOKEN` extends that MTP checkpoint through the appended
  block's self-token full attention, residual add, dense/MoE FFN, and final
  residual add. On the target GGUF this covers layer-24 Q2_K/Q3_K attention and
  dense FFN tensors after the Q8_0 NextN projection.
- `--qwen-mtp-draft-test TOKEN` creates a native Qwen session, runs the main
  graph for one committed token, saves the target hidden state, then executes
  MTP/NextN from `last_token_embd + main_hidden` through the appended block,
  shared/output norm, and logits head. The target smoke pins draft
  hidden/logits checksums and verifies the draft path is allocation-free under
  the native Qwen allocation guard.

### M1.5: Qwen Compatibility Generation Backend

Acceptance:

- Qwen GGUF one-shot generation can be cross-checked through the compatibility
  backend without changing the native DS4 build.
- Native Qwen one-shot generation remains the default CLI path.
- The target smoke test must prove path isolation: explicit
  `DS4_QWEN_COMPAT_GENERATE=1` emits the llama.cpp compatibility-backend log,
  while default, fast-dot, and MTP native generation do not emit that log.

Current command:

```sh
scripts/install_qwen_compat_backend.sh
DS4_QWEN_COMPAT_GENERATE=1 ./ds4 --metal \
  -m /Users/dinggh/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf \
  -p 'Say hello in one short sentence.' \
  -n 8 -c 1024 --temp 0 --nothink
```

The backend path can be overridden with:

```sh
export DS4_QWEN_LLAMA_CLI=/path/to/llama-cli
```

### M2: CPU Dense Qwen3.6-27B Correctness

Acceptance:

- CPU path loads dense model metadata and tensors.
- GDN, full attention, dense FFN, final logits pass golden tests.
- Token-by-token decode matches chunk prefill within threshold.

Current implementation:

- Native CPU reference scaffolding covers the target smoke model through:
  `token_embd.weight` Q6_K dequantization, `blk.0.attn_norm.weight` RMSNorm,
  `blk.0.attn_qkv.weight` Q4_K matvec, `blk.0.attn_gate.weight` IQ3_XXS
  matvec, `blk.0.ssm_out.weight` Q5_K matvec, dense `blk.0.ffn_gate.weight`
  Q2_K/IQ2_S/IQ3_S matvec, dense `blk.0.ffn_up.weight` Q2_K/IQ2_S/IQ3_S
  matvec, dense `blk.0.ffn_down.weight` Q3_K/IQ3_S/IQ4_XS matvec, and the dense FFN SwiGLU
  micro-path. It also covers the layer-0 recurrent GDN parameter micro-path
  through f32 conv/beta/alpha/dt/a tensors and a zero-state GDN update through
  gated RMSNorm plus `ssm_out`. The stateful GDN diagnostic now updates the
  layer-0 causal conv cache and recurrent state across two tokens, including
  the real buffers allocated by `ds4_session_create()`. The layer-0 zero-state
  residual diagnostic stitches the GDN output and dense FFN output together in
  decoder order, then validates final RMSNorm plus the tied Q6_K logits head for
  the target GGUF. The prefix-4 diagnostic executes layers 0-3 in decoder
  order, covering three recurrent GDN layers, the first full-attention layer,
  mixed Q2_K/Q3_K and IQ3_S/IQ4_XS dense FFN formats, final RMSNorm, and
  logits. Native `ds4_session_eval()` now runs prompt tokens through all 24
  main layers, including later full-attention Q2_K/Q3_K formats, later dense
  FFN IQ2_S/IQ3_S tensors, GDN state updates, and full-attention f16 KV cache
  updates. The first full-attention layer diagnostic covers the full-attention
  projection/output tensor formats used by the same GGUF.
- Checksums are fixed by `tests/qwen35_gguf_smoke.sh` with
  `--qwen-embed-test`, `--qwen-rms-test`, `--qwen-qkv-test`,
  `--qwen-attn-gate-test`, `--qwen-ssm-out-test`, `--qwen-gdn-param-test`,
  `--qwen-gdn-zero-test`, `--qwen-gdn-stateful-test`,
  `--qwen-session-step-test`, `--qwen-snapshot-test`, `--qwen-full-attn-test`,
  `--qwen-ffn-gate-test`, `--qwen-ffn-down-test`, `--qwen-ffn-test`,
  `--qwen-layer0-zero-test`, `--qwen-layer0-logits-test`, and
  `--qwen-prefix4-logits-test`. The same target GGUF smoke also pins native
  CLI `--dump-logits` and `--dump-logprobs -n 1` output, so both diagnostic
  file-dump paths are covered by the native Qwen session rather than only the
  sampled text path.
- The current M2 gate proves deterministic native decode on the target GGUF with
  exact `--dump-logprobs -n 1` assertions (`selected.id=361`,
  `logit=14.3216496`, `logprob=-0.57126087`) and default native generation path
  isolation. Remaining M2 expansion work is adding external reference golden
  vectors for Qwen3.6 dense once a local Qwen3.6 dense GGUF is available, then
  moving the CPU reference path to high-throughput Metal/CUDA kernels.
- Spark CUDA host validation on `ecs_spark_001` now covers real Qwen3.6-27B
  dense GGUFs under `~/models/Qwen3.6-27B-MTP-GGUF`. The Qwen3.6-27B-Q3_K_S
  inspector reports 64 main layers, 48 recurrent GDN layers, 16 full-attention
  layers, 1 MTP/NextN block, `embd=5120`, `heads=24`, `kv_heads=4`,
  `dt_rank=48`, and `groups=16`. This exposed two Qwen3.6-specific runtime
  gaps that are now fixed: GDN Q/K projections use grouped heads when
  `ssm_group_count != ssm_dt_rank`, and explicit `output.weight` may use any
  supported dense/quantized matvec format instead of being restricted to Q8_0.

### M3: CPU MoE Qwen3.6-35B-A3B Correctness

Acceptance:

- CPU path loads MoE tensor contract.
- Routed experts and shared expert match Transformers vectors.
- Full logits match golden tests.

Current implementation:

- The Qwen native CPU reference path now has a MoE FFN branch wired into the
  same decoder-layer FFN call site as dense SwiGLU.
- Native session decode, prefix diagnostics, layer-0 diagnostics, and MTP/NextN
  draft execution now call the shared Qwen FFN dispatcher, so dense checkpoints
  use dense SwiGLU while Qwen MoE checkpoints use routed/shared expert FFN at
  the same decoder/MTP call sites.
- The MoE branch supports router projection, softmax over all experts, top-k
  selection, top-k probability renormalization, separate or fused routed
  gate/up expert tensors, routed down experts, and optional shared expert
  gating with `sigmoid(shared_scalar_gate(x))`.
- Routed/shared expert matvecs use the generic Qwen tensor path for F32, F16,
  Q8_0, and the locally supported K/IQ quantized formats already used by dense
  Qwen smoke tests. Q8_0 is covered for normal 2-D projections, 1-D shared
  scalar gates, and 3-D routed expert slices.
- `--qwen-moe-ffn-test TOKEN` is available as the first MoE layer-0 golden-vector
  hook. The current dense smoke test verifies that the diagnostic rejects dense
  GGUFs, and the next MoE GGUF should pin its output checksum here before
  expanding to full-logits tests.
- Routed expert slices are evaluated row-parallel across F32/F16/Q8_0 and the
  supported K/IQ formats, so the CPU reference is not locked into a serial
  per-row expert scan while Metal/CUDA kernels are being built.
- In session decode, the MoE FFN branch reuses `qwen35_native_state` scratch for
  router logits/probabilities, top-k selections, expert weights, routed
  gate/up/mid/down buffers, and fused gate-up output. The standalone
  `--qwen-moe-ffn-test` diagnostic still owns local scratch so it remains
  independent of session allocation.
- This is still a CPU reference implementation. It compiles and does not affect
  the dense Qwen3.5 smoke GGUF, but it still needs an actual Qwen3.5/Qwen3.6 MoE
  GGUF plus Transformers/llama.cpp golden vectors before M3 can be marked
  complete.

### M4: Metal Decode

Acceptance:

- Dense and MoE decode run on Metal.
- Full-attention layers use the existing high-performance attention style where
  applicable.
- GDN single-token decode uses backend kernels rather than scalar CPU fallback.
- Correctness remains within threshold.

### M5: Metal Prefill

Acceptance:

- Chunked GDN prefill is implemented.
- Prefill throughput is benchmarked at 2k, 8k, and 32k prompts.
- Prefill state exactly seeds decode.

### M6: MoE Optimization

Acceptance:

- Router/top-k and expert batching avoid per-token expert loops.
- Qwen3.6-35B-A3B reaches an agreed local tok/s target.
- Quantized expert recipes have quality and speed reports.

### M7: MTP

Acceptance:

- MTP block loads and runs as a separate draft graph.
- Rollback restores KV and GDN states.
- Latency improvement is measured against throughput regression.

Current implementation:

- The Qwen loader binds MTP/NextN block tensors appended after the main decoder
  stack. The target Qwen3.5-0.8B MTP GGUF smoke asserts `blk.24` full-attention
  Q2_K/Q3_K tensors, dense FFN Q2_K/Q3_K tensors, `nextn.eh_proj` Q8_0, and
  `nextn.enorm/hnorm/shared_head_norm` F32 tensors through `--inspect`.
- `--qwen-mtp-proj-test TOKEN` executes the deterministic NextN projection
  micro-path and the target smoke pins token-0 checksum values. This verifies
  that the MTP Q8_0 projection is not merely present in metadata but usable by
  the native Qwen CPU reference helpers.
- `--qwen-mtp-layer-test TOKEN` then runs the projected hidden state through the
  appended MTP block's self-token full-attention and FFN path, pinning checksums
  for the projection, attention output, FFN output, and residual result.
- `--qwen-mtp-draft-test TOKEN` executes the appended block as a native Qwen
  draft-logits graph seeded by the main graph's saved hidden state and the last
  token embedding. This is still a single-token CPU reference drafter, but it
  proves the runtime wiring needed by speculative decode: target hidden capture,
  NextN projection, appended block execution, shared/output head, top-1 draft,
  and no hot-path allocation.
- `--mtp-draft N` now activates the embedded Qwen NextN drafter for greedy
  native Qwen CLI generation. The current implementation is correctness-safe
  and sequential: it only commits a draft token after the target logits agree,
  so it is useful as a runtime probe but not yet a latency win.
- Qwen MTP draft failures are non-fatal in normal session decode, matching the
  DeepSeek MTP contract: the target model token has already been committed, and
  a failed drafter simply disables that speculative step. `DS4_MTP_PROBE=1`
  logs Qwen draft hit/miss counters and is covered by the target GGUF smoke.
- M7 remains open until batched/parallel Qwen draft verification, rollback of
  Qwen GDN/full attention state, and latency measurements are implemented.

### M8: CUDA

Acceptance:

- CUDA backend implements fused GDN decode.
- CUDA backend implements chunked GDN prefill or documents a measured fallback.
- Dense and MoE pass the same correctness suite.

Current implementation:

- `make cuda-spark` builds on `ecs_spark_001` with CUDA 13.0 and NVIDIA GB10.
- Real Qwen3.6-27B dense GGUF native decode runs under the CUDA build. The
  measured path now initializes CUDA, registers the GGUF model map, prepares the
  Spark HBM tensor-span cache, and dispatches supported Qwen quantized matvecs
  to CUDA. It is not yet a fully device-resident fused GDN/FFN/logits path.
  Short benchmark command:
  `QWEN_BENCH_BACKEND=cuda QWEN_BENCH_RUNS=1 QWEN_BENCH_PROMPT_REPEATS=1
  QWEN_BENCH_GEN_TOKENS=1 QWEN_BENCH_CTX=1024 ./scripts/qwen35_bench.sh`.
- Initial `ecs_spark_001` Qwen3.6-27B reference-path results:

| Model | Backend | Prefill t/s | Gen t/s | Notes |
| --- | --- | ---: | ---: | --- |
| Qwen3.6-27B-Q3_K_S.gguf | cuda build | 0.47 | 0.46 | Native reference path |
| Qwen3.6-27B-Q3_K_M.gguf | cuda build | 0.52 | 0.52 | Native reference path |
| Qwen3.6-27B-Q4_1.gguf | cuda build | 0.42 | 0.43 | Native reference path |
| Qwen3.6-27B-IQ4_NL.gguf | cuda build | 0.49 | 0.50 | Native reference path |

- Current `ecs_spark_001` Qwen3.6-27B CUDA results, command shape:
  `env DS4_QWEN_COMPAT_GENERATE=0 DS4_QWEN_NATIVE_GENERATE=1 ./ds4 --cuda
  -m ~/models/Qwen3.6-27B-MTP-GGUF/<model> -p hi -n 16 -c 1024 --temp 0
  --nothink`.

| Model | CUDA coverage | Prefill t/s | Gen t/s | Baseline gen t/s | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.6-27B-Q3_K_S.gguf | Q3_K striped2 matvec + dense FFN + recurrent GDN + full-attention decode | 3.88 | 3.93 | 0.46 | Correct default path |
| Qwen3.6-27B-Q3_K_M.gguf | Q3_K striped2 matvec + dense FFN + recurrent GDN + full-attention decode | 1.49 | 1.49 | 0.52 | Correct default path |
| Qwen3.6-27B-Q4_1.gguf | Q4_1 striped-f32 matvec + dense FFN + recurrent GDN beta/alpha F32 pair + full-attention Q/K/V triple | 9.86 | 10.20 | 0.43 | Correct default path |
| Qwen3.6-27B-IQ4_NL.gguf | IQ4_NL striped-f32 matvec + dense FFN + recurrent GDN + full-attention decode | 5.42 | 5.53 | 0.50 | Correct default path |

- Same-host llama.cpp reference, built from current upstream with
  `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121` on the same NVIDIA GB10:
  `./build/bin/llama-bench -m
  ~/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_1.gguf -ngl -1 -fa on -p 23
  -n 16 -b 2048 -ub 2048 -r 3 -o jsonl`.

| Runtime | CUDA graphs | Prefill t/s | Decode t/s | Notes |
| --- | --- | ---: | ---: | --- |
| llama.cpp | default | 211.72 | 12.70 | `qwen35 27B Q4_1`, full GPU offload, FlashAttention |
| llama.cpp | `GGML_CUDA_DISABLE_GRAPHS=1` | 218.16 | 12.86 | Graphs are not the main delta for this short decode |
| ds4 native | default | 9.86 | 10.20 | Token-by-token prompt path, striped Q4_1 f32-activation matvecs plus GDN beta/alpha and full-attention Q/K/V pair/triple kernels |

- CUDA single-token matvec kernels are now wired for F32, Q3_K, Q4_1, Q5_K,
  IQ4_NL, and Q6_K. Q6_K is needed for tied/explicit output-head variants even
  though the current Spark Qwen3.6 dense GGUFs did not exercise Q6_K in the
  final measured profile. F32 and Q5_K are required for Qwen GDN beta/alpha and
  `ssm_out` projections. CUDA dispatch profiling confirmed zero fallback for
  Q3_K_M, Q4_1, and IQ4_NL. Earlier final-token profile data before F32/Q5_K:
  - Q3_K_M: 208 Q3_K calls/token, H2D 2.55 ms, kernel 1.66 ms, D2H 50.17 ms.
  - Q4_1: 352 Q4_1 calls/token, H2D 1.99 ms, kernel 1.29 ms, D2H 120.09 ms.
  - IQ4_NL: 288 IQ4_NL calls/token, H2D 2.84 ms, kernel 1.66 ms, D2H 217.09 ms.
  After the Q4_1 striped-f32 default, Q4_1 profiling reports 496 logical CUDA
  matvecs per token: 352 Q4_1, 96 F32, and 48 Q5_K. The measured matvec kernel
  enqueue section was about 1.9 ms/token before GDN beta/alpha pairing, while
  the CPU-visible synchronization point still accounted for about
  89-90 ms/token. This confirms the next large Q4_1 gain must reduce
  launch/synchronization count or fuse larger recurrent and FFN subgraphs;
  further single-kernel micro-optimizations on F32/Q5_K alone are unlikely to
  close the remaining llama.cpp gap.
  These numbers proved the first CPU/GPU ping-pong bottleneck. After the
  resident path removed most host transfers, llama.cpp still decodes about
  1.8x faster, so the remaining bottleneck is the CUDA kernel schedule itself:
  ds4 still uses simple f32-activation quant matvecs and many per-layer launches,
  while ggml uses tuned MMVQ/MMQ kernels and specialized Qwen recurrent kernels.
- Dense FFN now has a partial device-resident subgraph for CUDA-supported
  quantized tensors: RMSNorm is still computed on CPU, then the normalized input
  is written once to GPU, gate/up matvecs run on device, SwiGLU runs on device,
  the down projection runs on device, and only the final embedding-sized FFN
  output is read back. On Q4_1 this reduced final-token readback from about
  13.94 MiB/token to 5.44 MiB/token and raised generation from 3.06 to
  3.18 t/s in the short benchmark. The limited gain confirms the next major
  speedup must come from recurrent GDN and full-attention state staying on GPU.
- Recurrent GDN single-token decode now keeps the convolution cache and
  recurrent state on device by default for CUDA-supported tensor formats. It
  fuses the causal conv/cache update, Q/K L2 normalization, recurrent state
  update, gated RMSNorm, and `ssm_out` projection into the CUDA decode path.
  The previous correctness mismatch was caused by reusing the single global
  CUDA temp allocation for both GDN conv and output temporaries; the kernel now
  allocates one combined scratch span and slices it explicitly. `-p hi -n 16
  --temp 0 --nothink` matches the GDN-disabled path on all four Spark GGUFs.
  The fallback switch is `DS4_QWEN_DISABLE_CUDA_GDN=1`.
- Recurrent GDN beta/alpha projections now use a default F32 pair matvec when
  both tensors are F32 and have matching dimensions. The kernel preserves the
  original per-row warp reduction order for each projection but computes beta
  and alpha in one launch, removing 48 launches per Qwen3.6-27B token while
  keeping the profiling counters as two logical F32 matvec calls. Spark Q4_1
  short greedy validation matched the old path exactly (`stdout_cmp=0`).
  Profiled recurrent time improved from about `27.5 ms/token` to
  `26.4 ms/token`, and unprofiled alternating runs measured default
  `9.82-9.85/10.16-10.19 t/s` versus rollback
  `9.70-9.73/10.04-10.07 t/s`. The rollback switch is
  `DS4_QWEN_DISABLE_CUDA_GDN_BETA_ALPHA_PAIR=1`.
- Full-attention K/V projections now use a default Q4_1 pair matvec when both
  tensors are Q4_1 and have matching dimensions. This is intentionally scoped
  to the validated striped-f32 Q4_1 path and is disabled when the Q4_1 half16
  or activation-Q8 experiments are selected. It removes 16 launches per
  Qwen3.6-27B token on Q4_1. Spark short greedy validation matched rollback
  exactly (`stdout_cmp=0`); profiled full-attention time moved from about
  `7.17 ms/token` to about `7.0 ms/token`, and alternating unprofiled runs
  measured default `9.50-9.91/10.19-10.22 t/s` versus rollback
  `9.80-9.82/10.16-10.17 t/s`. The rollback switch is
  `DS4_QWEN_DISABLE_CUDA_FULL_KV_PAIR=1`.
- Full-attention Q/K/V projections now also have a Q4_1 triple matvec path that
  replaces the QG projection plus the K/V pair with one launch when all three
  tensors are Q4_1. The triple path keeps the validated striped-f32 reduction
  order for each projected row and falls back to K/V pair or scalar matvecs
  under the same experiment guards. Spark Q4_1 short validation matched
  rollback exactly (`stdout_cmp=0`). Profiled full-attention time moved from
  about `7.0 ms/token` to `6.85-6.96 ms/token`; unprofiled alternating runs
  measured default `9.84-9.87/10.18-10.22 t/s` versus rollback
  `9.51-9.84/10.17-10.19 t/s`. The gain is small, but launch count moves in
  the right direction. The rollback switch is
  `DS4_QWEN_DISABLE_CUDA_FULL_QKV_TRIPLE=1`.
- The GDN state-update kernel now defaults to the block32 launch shape. It keeps
  the original shared-memory reduction algorithm but uses 32 threads per
  `(head,value_column)` block instead of 256, matching Qwen's small state vector
  better. Spark Q4_1 logprob validation over 16 greedy steps matched the old
  kernel exactly (`selected_equal=true`, common top-logit max diff `0.0`), and
  `-n 32` throughput improved from `6.76/6.91 t/s` to `7.02/7.19 t/s`.
  The current default groups four value columns per block with the same
  shared-memory reduction order as block32; Spark Q4_1 validation matched
  exactly (`selected_equal=true`, common top-logit max diff `0.0`) and measured
  `7.02/7.19 t/s` vs standalone block32 `6.94/7.19 t/s` on `-n 32`.
  `DS4_QWEN_DISABLE_CUDA_GDN_SHARED4=1` restores standalone block32, and
  `DS4_QWEN_DISABLE_CUDA_GDN_BLOCK32=1` restores the old 256-thread launch. A
  more llama.cpp-like four-column warp-shuffle kernel remains experimental
  behind `DS4_QWEN_ENABLE_CUDA_GDN_WARP4=1`; it was faster (`7.04/7.21 t/s`)
  but failed greedy-token validation, so it must not become default without a
  reduction/layout fix.
- The GDN L2-normalization and gated-RMS kernels now default to 128 threads per
  block instead of 256. Qwen3.6-27B reports `state=128` and
  `value_head_dim=128`, so the old launch spent half its lanes on empty work.
  Spark Q4_1 logprob validation matched exactly against the old launch
  (`selected_equal=true`, common top-logit max diff `0.0`). The measured effect
  is small but non-negative (`6.96/7.19 t/s` old norm launch vs
  `7.01/7.19 t/s` default norm128 on `-n 32`). The rollback switch is
  `DS4_QWEN_DISABLE_CUDA_GDN_BLOCK128=1`.
- A wider eight-column GDN state kernel exists behind
  `DS4_QWEN_ENABLE_CUDA_GDN_SHARED8=1`. It keeps the exact shared-memory
  reduction order used by the validated block32/shared4 kernels, and Spark Q4_1
  JSONL logprob dumps matched the default path byte-for-byte (`cmp_rc=0`) over
  the short greedy validation. It did not improve throughput (`6.99/7.19 t/s`
  vs default `7.01/7.19 t/s` on `-n 32`), so shared4 remains the default. This
  bounds the remaining GDN state-launch win: simply grouping more value columns
  is not enough to close the llama.cpp decode gap.
- Full-attention single-token decode now has a Qwen-specific CUDA path for
  supported tensor formats. It keeps Q/K/V projection outputs on device, applies
  per-head Q/K RMSNorm, Qwen M-RoPE, f16 K/V cache update, causal GQA attention,
  attention gate, and output projection on CUDA. The normal fast path does not
  read K/V back to host per token; snapshot serialization synchronizes the full
  CUDA KV cache when needed. `DS4_QWEN_DISABLE_CUDA_FULL_ATTN=1` restores the
  CPU full-attention body, and `DS4_QWEN_SYNC_CUDA_FULL_KV=1` forces eager host
  KV synchronization for debugging.
- An experimental dense-only resident CUDA decode path exists behind
  `DS4_QWEN_ENABLE_CUDA_RESIDENT=1`. It keeps hidden, recurrent/full attention,
  residual adds, FFN, final norm, and logits on device across the main-layer
  loop. It is intentionally not default yet: after fixing GDN's resident
  head-count mapping to use `ssm_dt_rank` rather than Q attention heads, it runs
  through Q4_1 short greedy decode, but the measured short prompt is still
  about equal to the default path. After the block32/block128 GDN launch
  changes, Q4_1 `-n 32` still measured slightly below default
  (`6.97/7.14 t/s` resident vs `6.98/7.19 t/s` default).
- Resident profiling is now available with
  `DS4_QWEN_CUDA_RESIDENT_PROFILE=1`. On Spark Q4_1 it shows each token issuing
  497 CUDA matvecs with zero fallback and no host transfer inside the resident
  matvec path. The apparent `hidden_read` or `output` cost of about 132-140 ms
  is the final synchronization point for the full queued token workload, not a
  large activation copy. An opt-in greedy path
  `DS4_QWEN_CUDA_GREEDY_TOP_ONLY=1` computes GPU argmax and reads only the token
  id instead of full logits; it measured `6.94 t/s`, essentially unchanged from
  the resident full-logits path. The remaining bottleneck is therefore launch
  count and unfused per-layer work, not logits D2H.
- The same greedy top-only shortcut is now also wired into the default
  non-resident CUDA path behind `DS4_QWEN_CUDA_GREEDY_TOP_ONLY=1`. It writes the
  final hidden vector to CUDA, performs output RMSNorm, logits projection, and
  argmax on device, and reads only the token id. Spark Q4_1 output stayed
  unchanged, but generation did not improve (`6.94 t/s` vs clean default
  `6.95 t/s`, prefill `6.78 t/s` vs `6.61 t/s`). This confirms the default
  path's final logits D2H is also not the limiting factor.
- A narrower add+RMSNorm fusion exists as an experiment behind
  `DS4_QWEN_ENABLE_CUDA_ADD_RMS_FUSION=1`. It combines
  `after_attn = cur + attn` and the following FFN RMSNorm into one kernel. Spark
  Q4_1 profiling showed lower apparent per-section timings, but no generation
  gain (`6.92 t/s`) and worse prompt throughput in the profiled run
  (`4.11 t/s` vs `6.76 t/s` with the fusion disabled). Retesting after the
  striped-f32 matvec change showed it is now neutral for decode (`10.04-10.07
  t/s`) and output-identical on the short greedy check, but still not a clear
  win. Keep it opt-in; it proves that isolated one-kernel fusions are
  insufficient compared with a whole-token captured/fused schedule.
- A llama.cpp-inspired Q4_1 FFN `gate + up + SwiGLU` pair kernel exists behind
  `DS4_QWEN_ENABLE_CUDA_FFN_PAIR=1`. It was tested on Spark Q4_1 and is also
  kept opt-in: the naive fused row kernel reduced prompt throughput
  (`4.09 t/s` vs `6.79 t/s`) and did not improve decode (`6.95 t/s` vs
  `6.96 t/s`). This validates the design rule that ds4 should borrow ggml's
  MMVQ-style activation quantization and tiling, not just merge adjacent row
  kernels.
- A first Q4_1 activation-q8 experiment exists behind
  `DS4_QWEN_ENABLE_CUDA_Q4_1_Q8V=1`. It quantizes each activation vector to a
  q8_1-style block tensor and runs a Q4_1 x Q8_1 row kernel. This also remains
  opt-in only: after fixing the Q4_1 `m * sum(x)` accumulation bug, Spark Q4_1
  still produced visibly different greedy tokens and measured worse throughput
  (`3.94/6.49 t/s` vs clean default `6.71/6.94 t/s`). The failure mode is useful:
  a separate q8 quantization launch plus scalar lane math is not enough. A
  follow-up variant replaced the scalar lane inner loop with packed `dp4a`, but
  remained incorrect for greedy output and became slower (`3.05/4.35 t/s`).
  The production path must use ggml-like vectorized MMVQ tiling and must pass a
  logits acceptance gate before it can replace the f32-activation Q4_1 kernel;
  simply wrapping ds4's row-per-warp schedule with q8_1 is counterproductive.
- The Q4_1 activation-q8 path now has a striped dp4a variant that distributes
  eight Q4_1 blocks across all 32 lanes of the row warp. It is the first
  Q4_1 experiment that moves performance in the right direction: Spark Q4_1
  `-n 32` measured about `9.68/10.02 t/s`, versus the clean default
  `6.98/7.18 t/s` and the old Q8V row kernel `4.38/4.42 t/s`. It is still not
  production-correct. Full Q8V changes the short greedy top token from default
  `59` to `79320`; FFN-only, gate/up-only, and down-only splits also fail the
  top-1 acceptance gate over the same validation. A reconstructed-sum q8_1
  variant (`DS4_QWEN_CUDA_Q8_1_RECON_SUM=1`, storing `d * sum(qs)` rather than
  the original float sum) changes the failure mode but does not restore default
  top-1. Keep these switches opt-in:
  `DS4_QWEN_ENABLE_CUDA_Q4_1_Q8V=1` for full Q4_1,
  `DS4_QWEN_ENABLE_CUDA_Q4_1_Q8V_FFN=1` for FFN-only,
  `DS4_QWEN_ENABLE_CUDA_Q4_1_Q8V_FFN_GATEUP=1` for gate/up,
  `DS4_QWEN_ENABLE_CUDA_Q4_1_Q8V_FFN_DOWN=1` for down, and
  `DS4_QWEN_DISABLE_CUDA_Q4_1_Q8V_STRIPED=1` to restore the old Q8V row kernel.
  The result is still valuable: it proves GB10 wants the MMVQ lane/block
  schedule, but ds4 needs a logits-acceptable q8 quantization/fusion design
  before this can become the default high-performance path.
- The same lane/block schedule now exists for Q4_1 with f32 activations and is
  the default Q4_1 CUDA matvec. It keeps activation precision unchanged while
  distributing eight Q4_1 blocks across all 32 lanes of the row warp. Spark
  Q4_1 short greedy validation matched the old default top-1 for all 16 checked
  steps (`top1_match=16/16`); common top-logit max difference was about
  `0.00125`, caused by the changed reduction order. Throughput improved from
  the old kernel's `7.01/7.18 t/s` to `9.72-9.73/10.07 t/s` on `-n 32`,
  closing most of the gap to llama.cpp's `12.70 t/s` short decode. The rollback
  switch is `DS4_QWEN_DISABLE_CUDA_Q4_1_STRIPED_F32=1`. The older half-warp
  experiment remains available with `DS4_QWEN_ENABLE_CUDA_Q4_1_HALF16=1`.
- A wider Q4_1 f32 striped16 experiment exists behind
  `DS4_QWEN_ENABLE_CUDA_Q4_1_STRIPED16_F32=1`. It maps sixteen Q4_1 blocks
  across the row warp, but this made the schedule slower on NVIDIA GB10 despite
  matching the default greedy output (`stdout_cmp=0`): Spark Q4_1 profiled
  generation fell to about `7.85 t/s`, with FFN/recurrent synchronization time
  increasing sharply. Keep striped8 as the default; striped16 is a documented
  negative result.
- IQ4_NL now uses the same eight-block striped f32-activation row schedule by
  default. Spark IQ4_NL short greedy validation matched the old kernel top-1
  for all 16 checked steps (`top1_match=16/16`) with common top-logit max diff
  about `0.00062`. Throughput improved from the old kernel's `3.91-3.96 t/s`
  to `5.39-5.42/5.49-5.53 t/s` on `-n 32`. The rollback switch is
  `DS4_QWEN_DISABLE_CUDA_IQ4_NL_STRIPED=1`.
- Q3_K now has a two-superblock striped row kernel by default. It uses both
  half-warps per row to process two Q3_K superblocks at a time while preserving
  f32 activations. Spark Q3_K_S and Q3_K_M short greedy validation matched the
  old kernel top-1 for all 8 checked steps; common top-logit max diff stayed
  below about `0.0011`. The throughput effect is positive but small:
  Q3_K_S improved from `3.80/3.85 t/s` to `3.88/3.93 t/s`, and Q3_K_M improved
  from `1.47/1.45 t/s` to `1.49/1.49 t/s`. The rollback switch is
  `DS4_QWEN_DISABLE_CUDA_Q3_K_STRIPED=1`.
- A four-superblock Q3_K striped variant exists behind
  `DS4_QWEN_ENABLE_CUDA_Q3_K_STRIPED4=1`. It also passed the Spark short top-1
  validation for Q3_K_S and Q3_K_M, but the throughput tradeoff is
  model-dependent: Q3_K_S improved in one run (`3.90` to `4.12 t/s` decode),
  while Q3_K_M slightly regressed (`1.47` to `1.45 t/s`). Keep striped2 as the
  default until ds4 can select Q3_K kernels from stronger per-model profiling
  evidence.
- A striped Q4_1 FFN `gate + up + SwiGLU` pair kernel exists behind
  `DS4_QWEN_ENABLE_CUDA_FFN_PAIR=1` and can be forced back to the old pair
  kernel with `DS4_QWEN_DISABLE_CUDA_FFN_PAIR_STRIPED=1`. It matches the
  current default logits exactly on the Spark Q4_1 short validation, but it did
  not improve `-n 32` throughput (`10.05-10.06 t/s` either way), so it remains
  opt-in.
- A Q4_1 half-warp matvec experiment exists behind
  `DS4_QWEN_ENABLE_CUDA_Q4_1_HALF16=1`. It maps one row to 16 lanes instead of a
  full warp, so a 256-thread block covers 16 rows rather than 8. Spark Q4_1
  logprob validation matched the default exactly (`selected_equal=true`, common
  top-logit max diff `0.0`), but throughput regressed (`6.74/6.90 t/s` vs
  default warp8 `7.02/7.19 t/s` on `-n 32`). Keep it opt-in; on GB10 the current
  row-per-warp schedule is faster despite half-warp lane under-utilization.
- llama.cpp implementation details that are directly relevant:
  - `src/models/qwen35.cpp` builds Qwen3.5/3.6 as a hybrid DeltaNet model,
    routes recurrent layers through `llm_build_delta_net_base`, uses Q/K norms,
    M-RoPE, FlashAttention-capable full-attention layers, and marks MTP layers
    as non-recurrent.
  - `ggml/src/ggml-cuda/gated_delta_net.cu` runs GDN recurrence with a
    warp/column register-sharded kernel over tokens, heads, sequence slots, and
    value columns. ds4's current GDN path is single-token and split across more
    kernels.
  - `ggml/src/ggml-cuda/mmvq.cu` quantizes activations to q8_1 blocks and uses
    typed vector-dot kernels, including gate/GLU fusion. ds4 currently dots raw
    f32 activations against GGUF rows.
  - `benches/dgx-spark/dgx-spark.md` uses GB10, full GPU offload,
    FlashAttention, and large `n_batch`/`n_ubatch`; ds4's native prefill remains
    token-by-token and therefore cannot match the 200+ t/s prompt number yet.
- M8 remains open. These measurements are useful because they prove real
  Qwen3.6 dense model compatibility on Spark and quantify the baseline, but they
  also show that performance work must next target ggml-class projection/GDN
  kernels and chunked prefill before CUDA Graph capture is expected to matter.
- The DeepSeek V4 CUDA path is the implementation reference for the next Qwen
  work, especially:
  - `ds4_gpu_tensor` device-owned buffers and command batching so decode does
    not bounce activations through host memory.
  - model-map registration/spans and Spark HBM cache policy for hot tensors.
  - quantized matvec entry points and fused pair projections for gate/up style
    FFN work.
  - decode/prefill attention kernels that keep KV state device-resident.
  - GPU argmax/top-k plumbing for sampler variants; this is useful for API
    shape, but Qwen resident measurements show it is not the primary speed
    limiter by itself.
- Qwen3.6 Spark optimization order:
  1. Port an MMVQ-style CUDA matvec path for Q4_1 first: q8_1 activation
     quantization, tuned row/block scheduling for GB10, and fused gate/up GLU
     using ggml's `mmvq.cu` design rather than ds4's naive row-pair or scalar
     Q8V kernels. Acceptance requires matching greedy output on the Spark Q4_1
     smoke prompt and improving decode throughput over the clean default.
  2. Replace the current split GDN decode with a Qwen Gated DeltaNet kernel
     shaped like ggml's `gated_delta_net.cu`, then extend it to multi-token
     prompt chunks.
  3. Add chunked native prefill (`n_batch`/`n_ubatch` equivalent) for recurrent
     and full-attention layers. This is required to close the 6.8 vs 211 t/s
     prefill gap.
  4. Keep the Qwen GPU session state holding hidden/norm/qkv/gate/ffn/logits,
     full-attention KV, residuals, and final sampler workspace on device. The
     recurrent GDN state and convolution cache already live on device in the
     CUDA decode path.
  5. Evaluate CUDA Graph capture only after the matvec/GDN schedule resembles
     llama.cpp; same-host llama.cpp data with graphs disabled did not regress.
  6. Only after dense Qwen3.6 reaches an agreed Spark target should MoE expert
     batching/offload be implemented.

### M9: Large Qwen3.5 MoE

Acceptance:

- Qwen3.5-397B-A17B metadata and tensor validation pass.
- Expert paging/offload or multi-GPU expert-parallel strategy is implemented.
- Performance and memory behavior are documented.

## 11. Engineering Decisions

### Decision: Separate Runtime Family

Do not overload the existing DeepSeek V4 `ds4_shape` path. Qwen3.5/3.6 has
different attention, cache, tensor, and tokenizer contracts. A shared low-level
matmul/quant/kernel library is fine, but model-family validation and execution
must be separate.

### Decision: Hybrid Cache Is Mandatory

Treating all layers as KV attention is incorrect and too expensive. Treating
GDN state as ordinary KV is also wrong. The runtime must allocate full KV only
for full-attention layers and recurrent state only for GDN layers.

### Decision: Converter Owns Layout Fixups

The runtime should not pay per-token costs for `A_log` transforms, fused
projection splitting, or V-head reorder. These are deterministic conversion
steps.

### Decision: MoE Must Be Performance-First

Qwen MoE support without expert batching, quantization strategy, and shared
expert fast path is not acceptable. It may be correct, but it will not meet the
project goal.

## 12. Source References

- Hugging Face Qwen3.5/Qwen3.6 model documentation:
  https://huggingface.co/docs/transformers/model_doc/qwen3_5
- NVIDIA NeMo/Megatron Bridge Qwen 3.5/3.6 documentation:
  https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/vlm/qwen35-vl.html
- Qwen3.6-35B-A3B config:
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json
- Qwen3.6-27B config:
  https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json
- Qwen3.6 tokenizer config:
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/tokenizer_config.json
- Transformers Qwen3.5 dense implementation:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Transformers Qwen3.5 MoE implementation:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
- vLLM Qwen3.5/3.6 usage guide:
  https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md
- SGLang Qwen3.6 deployment guide:
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6
- SGLang Qwen3.5 deployment guide:
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5
- vLLM Qwen3.5 model implementation:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py
- vLLM Gated DeltaNet implementation:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- vLLM Mamba/GDN state utilities:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py
- llama.cpp Qwen3.5 dense implementation:
  https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- llama.cpp Qwen3.5 MoE implementation:
  https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp
- llama.cpp DeltaNet base:
  https://github.com/ggml-org/llama.cpp/blob/master/src/models/delta-net-base.cpp
- llama.cpp Qwen conversion:
  https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py
- ggml CUDA gated-delta-net kernel:
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu
- ggml CUDA MMVQ quant matvec kernel:
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu
- llama.cpp CUDA backend and graph controls:
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu
- llama.cpp DGX Spark benchmark notes:
  https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md
- vLLM Qwen3.5 FLA tensor-format issue:
  https://github.com/vllm-project/vllm/issues/38643
- vLLM hybrid GDN/attention FP8 KV-scale issue:
  https://github.com/vllm-project/vllm/issues/37554
- llama.cpp qwen35moe prompt/KV reuse issue:
  https://github.com/ggml-org/llama.cpp/issues/19690
