# IQ routed-expert F16 prefill qualification

Baseline: `0817d23b697460626d9da2bcb815b1a14ac1922c`, native Windows
gfx1151, TheRock 10.0.0 / clang 23. This is an experimental implementation
record, not a claim that the new arithmetic is qualified.

## Scope and sequence

The local UD-IQ4_XS checkpoint has 43 IQ3_S/IQ4_NL gate-up/down layer
combinations, four IQ3_S/Q8_0 combinations and one IQ4_XS/Q8_0 combination.
All 48 layers currently miss the routed F16 dispatch. The fallback already
uses INT8 WMMA; this work reuses the compact F16 expert pipeline and removes
its need for repeated routed activation quantization.

1. Adapt [IQ F16 prefill support](https://github.com/johandevlabs/gufo-iq4_xs/commit/bd13bae28c63b0b8579a4e2672e6d0a52804e582).
   Decode IQ codes to signed bytes during LDS staging, then reuse the Q8-style
   F16 fragment conversion and matrix loop. Stage the 512-entry IQ3_S grid
   in 2 KiB of LDS. Keep IQ gate/up separate initially.
2. Adapt [paired IQ gate/up](https://github.com/johandevlabs/gufo-iq4_xs/commit/88cd08069db1d6058f3a95816279c185ae765eb4).
   Require agreement with the separate F16 projections before measuring it.
3. Capture matched full-model logits/perplexity and state checks, then measure
   production throughput and matched HTTP latency. Keep Q4_K_XL as a control.

Keep the fork's existing Q4/Q5 cache selection, tile maps, decode kernels,
dense-weight conversion, model artifacts and toolchain. No full #570 import,
persistent expanded expert copy, new runtime switch or expert-count change.

## Acceptance criteria fixed before candidate results

- Independent operators use the existing `routed_wmma_ops` F64 reference and
  its unchanged limits: F16 error below 0.002 times the reference scale;
  MMQ error and F16/MMQ difference below 0.01 times that scale. Exercise the
  real 2560/640 dimensions, ragged output rows, 16/48/64-token tiles and
  1/8/9/32/33-token batches. CPU IQ decoders provide independent weight values.
- Paired 64/128-token gate/up tiles retain each separate F16 result, including
  subnormals, under the existing signed-zero convention. Replays are byte exact.
- Existing Q4_K_XL raw-logit/perplexity comparators stay exact. No golden,
  numerical tolerance, state assertion or model qualification gate is relaxed.
- IQ prefill intentionally replaces Q8 activation quantization with F16
  rounding. Record complete matched-token differences and perplexity; do not
  describe an expected hash mismatch as an exact qualification pass. The
  current `qualify.json` has exact comparators and no model-owned numerical
  limits for this transition. Independent CPU-reference comparisons and a
  separately justified numerical policy are required before promotion.
- Session chunking, private batching, sampling/rejection, rollback, snapshots
  and EOS must still pass. A changed prefill state does not excuse a replay
  inconsistency within the candidate.
- Performance results need the same artifacts, corpus, context, sampling,
  speculation and toolchain, warmed shapes and reversed baseline/candidate
  order. Separate native throughput from HTTP latency and startup time.
- Linux checks remain required for shared kernel code. Missing or unrun checks
  are explicit gaps, not passes.

## Evidence location

Raw builds and iteration logs: ignored `build/iq-prefill/`.
Raw model identities, captures and qualification records: ignored
`artifacts/iq-prefill/`. Results will be recorded here as checks complete.

## Stage 1: separate IQ F16 projections

- Fresh Windows GPU build succeeds. Expanded `qwen38_flash_next.routed_wmma_ops`
  passes in 83.26 seconds, including 24 IQ cases with unchanged F64/MMQ limits.
- All 23 compiled routed variants have zero scratch, VGPR spills and SGPR
  spills. The resource guard now requires the nine new separate IQ variants;
  its ten parser/contract fixtures pass.
- All 512 device IQ3_S grid entries match the independent core CPU table.
  Formatting (clang-format 21.1.8), docs and dependency checks pass.
- Three baseline CPU-oracle fixtures retain 52 complete GPU vocabulary rows:
  51/52 top-1 matches, mean KL 0.00439308 and maximum KL 0.03179 (printed
  per-row values have five decimal places). These are baseline measurements,
  not candidate quality evidence. Paired kernels and model qualification are
  still pending.
