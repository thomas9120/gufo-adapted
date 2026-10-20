# IQ routed-expert prefill gap — Qwen3.8 Flash-Next (2026-10-19)

Fork: `windows-port` at `0817d23b` (measured binary `e60bda5ec32c`). Model: `qwen4exp`,
48 blocks, 512 experts, 10 used, `embedding_length` 2560, `expert_feed_forward_length` 640.

Scope: the unsloth `UD-IQ4_XS` target prefills markedly slower than the larger
`UD-Q4_K_XL` target, even though it is the smaller file and is not slower at
decoding. This note records the measurement, the mechanism, and what closing the
gap would take. Decode-side findings are out of scope here.

## 1. Summary

- The gap is **prefill only**. Decode is at parity (AR) or ahead (MTP).
- Cause: the routed F16/WMMA matrix-core prefill GEMM accepts only
  `Q4_K`/`Q5_K` gate-up and `Q5_1`/`Q8_0` down weights. `UD-Q4_K_XL` satisfies
  that on **48 of 48** layers; `UD-IQ4_XS` satisfies it on **0 of 48** and falls
  back to the vendored llama.cpp int8-dot MMQ path.
- The two checkpoints have *identical* per-layer expert layout; only the weight
  formats differ. The dynamic quantizer picked IQ formats for the same tensors
  `UD-Q4_K_XL` put in K-quants.
- Upstream `gufo-org/gufo` main carries the identical restriction and cannot
  load `UD-IQ4_XS` at all. The candidate upstream implementation is PR #570,
  closed unmerged on 2026-10-15.

## 2. Measurements

`gufo bench`, release binary `e60bda5ec32c`, gfx1151 (Radeon 8060S, 20 CUs),
depth 0, temperature 0, three repetitions per point, AR unless stated. Loading
is excluded.

| Workload | `UD-IQ4_XS` | `UD-Q4_K_XL` | `Q4_K_XL` gain |
| --- | ---: | ---: | ---: |
| pp128 | 430.04 +/- 0.62 | 526.62 +/- 2.66 | +22.5% |
| pp512 | 752.79 +/- 2.78 | 1060.39 +/- 6.85 | +40.9% |
| pp2048 | 998.21 +/- 0.46 | 1598.63 +/- 3.14 | +60.2% |
| AR tg64 | 28.41 +/- 0.10 | 28.51 +/- 0.26 | +0.4% |
| AR tg128 | 27.94 +/- 0.07 | 28.37 +/- 0.09 | +1.5% |
| MTP-7 tg128 | 47.92 +/- 0.50 | 45.62 +/- 0.49 | -5.0% (IQ ahead) |
| model load to first token | 19.7 s | 23.4 s | IQ ahead |

The prefill gap widens with prompt length, which is the signature of a
throughput-limited routed GEMM rather than a fixed per-request cost.

End-to-end over HTTP, one fresh server per configuration, cold prompt of about
1.5K tokens, 128 generated tokens, greedy:

| Configuration | server ready | cold TTFT | total | generation tok/s |
| --- | ---: | ---: | ---: | ---: |
| `UD-IQ4_XS`, `--mtp-draft-vocab full` | 22.2 s | 2.86 s | 5.39 s | 50.6 |
| `UD-Q4_K_XL`, `--mtp-draft-vocab prefix` | 25.7 s | 2.07 s | 4.54 s | 51.8 |

Cold TTFT is 38% worse on the IQ target. (The two rows use different
`--mtp-draft-vocab` settings; that setting is a separate, decode-side finding
and is not part of this note.)

These numbers reproduce the fork's own recorded October 7 control pair in
[the Flash-Next benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md#windows-clang-23-prefill-regression-check):
`UD-IQ4_XS` AR pp2048 at 979-985 tok/s against `UD-Q4_K_XL` at 1594 tok/s, with
AR tg128 at 27.17 against 27.63 tok/s.

## 3. Root cause

### 3.1 The gate

`src/models/qwen38_flash_next/kernels/rocm/executor.cpp`, in
`Executor::MoeExperts`:

```cpp
const bool wmma_experts = ExpertMatrixRows(n_tokens) &&
                          (l.ffn_gate_exps.type == GgmlType::kQ4_K ||
                           l.ffn_gate_exps.type == GgmlType::kQ5_K) &&
                          l.ffn_up_exps.type == l.ffn_gate_exps.type &&
                          (l.ffn_down_exps.type == GgmlType::kQ5_1 ||
                           l.ffn_down_exps.type == GgmlType::kQ8_0) &&
                          c.hidden_size % 256 == 0 && c.expert_ff % 64 == 0;
```

`ExpertMatrixRows(n_tokens)` is `prefill_phase || rows > 4 * kVecBatch`
(`kVecBatch` is 8), so this gate is reached for any prefill chunk above 32
tokens.

When it holds, the layer takes the routed F16 path: `RoutedCompact` groups
token rows by expert into padded tiles, `NarrowActivations` converts the token
rows to F16 once, and `RoutedF16Gemm` /
`RoutedGatedF16Gemm` (`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`)
dequantize each expert weight tile to F16 and run it on the matrix cores
(`wmma_f32_16x16x16_f16_w32`). Each weight tile is decoded once and reused
across every token routed to that expert.

When it does not hold, the layer falls back to the vendored llama.cpp path:
`qfn_mmq_iq3_s_moe_raw` / `qfn_mmq_iq4_xs_moe_raw` / `qfn_mmq_iq4_nl_moe_raw`
(`src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp`), which is the
integer `dp4a` MMQ kernel. MMQ is tuned for small batches; at prefill widths it
leaves a large part of the matrix cores idle.

### 3.2 Why the IQ target never qualifies

Expert formats per layer, read from the GGUF tensor directories:

| layers | `UD-IQ4_XS` gate/up | `UD-IQ4_XS` down | `UD-Q4_K_XL` gate/up | `UD-Q4_K_XL` down | `Q4_K_XL` qualifies |
| --- | --- | --- | --- | --- | --- |
| 43 | `IQ3_S` | `IQ4_NL` | `Q4_K` | `Q5_1` | yes |
| 4 | `IQ3_S` | `Q8_0` | `Q4_K` | `Q8_0` | yes |
| 1 | `IQ4_XS` | `Q8_0` | `Q5_K` | `Q8_0` | yes |

The `UD-IQ4_XS` gate/up weights are `IQ3_S` or `IQ4_XS` on every layer, so the
first clause of the gate fails on all 48. The `UD-Q4_K_XL` target satisfies
every clause on all 48.

The restriction is structural, not a missing branch: `WeightType`
(`src/models/qwen38_flash_next/kernels/rocm/kernels.hpp`) enumerates only
`kF32`, `kF16`, `kBF16`, `kQ5_1`, `kQ8_0`, `kQ4_K` and `kQ5_K`, and both
`RowElement` and `RowBytes` in `kernels.hip.cpp` handle only `kF32`, `kBF16`,
`kF16` and `kQ8_0`. There is no IQ block reader on the routed F16 path at all.

### 3.3 Why decode is not affected

Decode never reaches `RoutedF16Gemm`. `ExpertMatrixRows` is false for decode
and for MTP verification at up to 32 tokens, so both targets run the vector
MMVQ path (`mul_mat_vec_moe_dispatch` in
`src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp`), which does carry
`IQ3_S`, `IQ4_XS` and `IQ4_NL` entries.

The IQ target is in fact the cheaper one at decode. Expert weight bytes read
per token, computed from the block geometries at 10 used experts:

| | gate/up | down | total per token |
| --- | --- | --- | ---: |
| `UD-IQ4_XS` | `IQ3_S` 110 B / 256 w | `IQ4_NL` 18 B / 32 w | 1162 MB |
| `UD-Q4_K_XL` | `Q4_K` 144 B / 256 w | `Q5_1` 24 B / 32 w | 1504 MB |

`UD-IQ4_XS` reads 23% fewer expert bytes per token, which is why its MTP decode
is ahead of the larger file.

### 3.4 Device footprint

Computed from the same tensor inventory, after the upload-time re-encoding to
Q8_0 that `device_model.cpp` performs for dense projections:

| | experts | dense | embedding | output head | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `UD-IQ4_XS` | 69.78 GiB | 3.83 GiB | 0.63 GiB | 0.63 GiB | 74.87 GiB |
| `UD-Q4_K_XL` | 71.73 GiB | 3.83 GiB | 0.63 GiB | 0.63 GiB | 76.82 GiB |

`per_layer_token_embd.weight` (26.82 GiB of `IQ4_NL` in both checkpoints) is
read from disk only and is not resident on the device
(`src/models/qwen38_flash_next/weights.hpp`). Both targets fit a 96 GiB
carve-out with room for the KV cache.

## 4. What a fix has to touch

Scope for a routed F16 prefill path that accepts the IQ formats.

1. **`WeightType`** (`src/models/qwen38_flash_next/kernels/rocm/kernels.hpp`):
   add `kIQ3_S`, `kIQ4_XS`, `kIQ4_NL`.
2. **Block readers** (`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`,
   `RowElement` and `RowBytes`): dequantize an IQ block to F16 per element, or
   add a block-level reader that fills an F16 tile directly. The block layouts
   are in `src/models/qwen38_flash_next/kernels/rocm/mmq/ggml-common.h`:
   - `block_iq3_s`: fp16 `d`, `qs[QK_K/4]` 3-bit codes, `qh[QK_K/32]`,
     `signs[QK_K/8]`, `scales[IQ3S_N_SCALE]`; 110 B per 256 weights.
   - `block_iq4_xs`: fp16 `d`, `scales_h`, `scales_l[QK_K/64]`, `qs[QK_K/2]`;
     136 B per 256 weights.
   - `block_iq4_nl`: fp16 `d`, `qs[QK4_NL/2]`; 18 B per 32 weights.
3. **The IQ3_S codebook.** `GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)` in
   `ggml-common.h` holds the eight-value codebook, four packed 4-bit values per
   entry. The MMQ kernel reads it as a scattered load
   (`vecdotq.hpp`, `vec_dot_iq3_s_q8_1`); a matrix-core path wants it gathered
   into LDS once per tile rather than per element. Upstream #570 describes this
   as "IQ3_S codebook gathered from LDS, one arithmetic at every batch width",
   and that batching-width invariance is the property to preserve.
4. **The gate** (`executor.cpp`): widen the `wmma_experts` clause to the IQ
   types, and widen `RoutedF16Gemm`'s `block_elems` selection (currently 256 for
   `Q4_K`/`Q5_K` and 64 otherwise) for the IQ block sizes. `IQ4_NL` is a
   32-weight block, so the `k` divisibility and tile-row assumptions need
   rechecking, not just a new enum value.
5. **The down projection.** `UD-IQ4_XS` uses `IQ4_NL` down weights on 43 of 48
   layers and `Q8_0` on the rest; the two formats coexist within one model, so
   the down path has to dispatch per layer rather than per model.

An alternative worth weighing before writing kernels: re-quantize the IQ
checkpoint's experts into `Q4_K`/`Q5_1` so it qualifies for the existing path.
That trades the memory saving for the prefill rate and produces a different
checkpoint, so it is not a code change but it is a legitimate answer to "why is
this file slow".

## 5. Upstream reference

- `gufo-org/gufo` main, `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`,
  carries the same `wmma_experts` clause, and main cannot load `UD-IQ4_XS` at
  all (`tensor blk.0.ffn_gate_exps.weight has unsupported format IQ3_S`).
- PR #570, "feat(qwen38-flash-next): add UD-IQ4_XS support, faster prefill and
  token-history drafting" (opened 2026-10-12, closed unmerged 2026-10-15), is
  the candidate implementation. It adds IQ routed prefill kernels in
  `kernels.hip.cpp` built on the wide-prefill routed kernels from #571, which
  this fork already carries as commit `a33e989c`; it also adds decode-side work
  (dense projections from a `Q5_1` copy of the `Q8_0` weights, attention
  compaction, token-history drafting) that is out of scope for this note. Its
  author reported about 1,694 tok/s at pp16384 on `UD-IQ4_XS`.
- #570 was closed rather than merged and this fork diverges from upstream in
  the Flash-Next kernels, so it is a reference to review, not a commit to
  cherry-pick. See [upstream-decode-flash-next.md](upstream-decode-flash-next.md)
  for the fork's standing on upstream PRs.

## 6. Validation requirements

Per [AGENTS.md](AGENTS.md) and
[matched-token validation](docs/TESTING.md#matched-token-and-layer-comparisons):

- Extend `tests/models/qwen38_flash_next/routed_wmma_ops_test.cpp` to the IQ
  formats at several tile sizes, comparing against the floating-point
  reference.
- Extend `tests/models/qwen38_flash_next/iq_experts_ops_test.cpp` to the routed
  F16 path, not just the vector and tiled MMQ paths it covers today.
- Matched-token full-logit equality and perplexity against a recorded
  `UD-IQ4_XS` baseline. A change to the routed prefill arithmetic is
  numerically visible, so require exact full-logit equality when the
  arithmetic should be unchanged, and declare limits if it intentionally moves.
- Paired ABBA prefill measurements at matched context and depth, with the build
  identity, model hashes, quantization, MTP sidecar, projector, context,
  sampling and speculation settings recorded, per
  [the model benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md).
- A separate pass for `UD-Q4_K_XL` to confirm the widened gate does not
  perturb the format that already qualifies.
- Linux evidence. The change touches shared kernel code, so the shared CI
  check and the Linux GPU path still apply.

## 7. Open questions

- Does the `IQ4_NL` 32-weight block survive the `k` divisibility and tile-row
  assumptions in the routed F16 path, or does it need a different down-projection
  tiling from `Q5_1`? This is the largest unknown and should be settled before
  the `IQ4_XS` gate/up work.
- The `IQ3_S` codebook gather is the hot path. Whether it lands in LDS per tile
  or per warp decides whether the path reaches matrix-core parity or only
  closes part of the gap.
- `UD-IQ4_XS` carries two expert formats per model role (gate/up and down),
  and some layers differ from others. The `wmma_experts` flag is per layer
  today, which is the right granularity, but a mixed model may want a
  per-projection decision instead of one flag for the layer.
- Whether the 23% lower expert traffic per token can be converted into decode
  headroom by raising the expert count, or whether the bandwidth is already
  saturated at the current count.
