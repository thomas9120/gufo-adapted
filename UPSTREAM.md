# Upstream adaptations

Track changes imported or adapted from other Gufo repositories here, with one
entry per logical improvement. Identify sources by repository URL and immutable
commit SHA; remote names alone are insufficient. This fork uses `official` for
`gufo-org/gufo` and `upstream` for `pixmaate/gufo`, subject to local configuration.

## Updating this record

- Add an entry when an upstream change is integrated, in the same change as the
  implementation. Reviewed, deferred, or skipped candidates are not imports;
  record those decisions under "Reviewed, not imported" below.
- Link every source commit and record the adaptation, omitted parts and reasons,
  affected areas, and validation actually performed. State failed or unrun checks
  explicitly; importing code does not establish that it passed validation.
- Follow [AGENTS.md](AGENTS.md) for `git cherry-pick -x` and `Upstream-Commit:`
  trailers. Once local implementation commits exist, record their SHAs. If this
  entry is committed with the implementation, identify that commit as "commit
  introducing this entry"; do not amend repeatedly to insert its own SHA.
- Keep entries newest first. Update status when a change is superseded or removed,
  linking the replacement or removal commit and retaining the original provenance.
- Link model-specific upstream contracts when relevant, without duplicating them.

Use this entry format:

```markdown
### YYYY-MM-DD — Improvement

- Source: [repository@full-SHA](https://github.com/owner/gufo/commit/full-SHA)
  (list every source commit, including imported prerequisites).
- Local: implementation commit SHA(s), or "commit introducing this entry".
- Adaptation: what was imported, changed or omitted, and why.
- Areas: affected paths; links to detailed model contracts when applicable.
- Validation: checks and results; failed or unrun checks and remaining limits.
- Status: retained / superseded / removed (with replacement or removal commit).
```

## Recorded adaptations

### 2026-10-19 — IQ routed-expert F16 prefill candidate

- Source: [johandevlabs/gufo-iq4_xs@bd13bae28c63b0b8579a4e2672e6d0a52804e582](https://github.com/johandevlabs/gufo-iq4_xs/commit/bd13bae28c63b0b8579a4e2672e6d0a52804e582),
  adapted.
- Local: commit introducing this entry.
- Adaptation: add IQ3_S/IQ4_XS gate/up and IQ4_NL down to the existing
  compact F16 routed kernels. Decode codes once in LDS staging; retain the
  fork's Q4/Q5 cache selection, tile maps and decode dispatch. The initial
  candidate keeps IQ gate/up separate. No loader, dense-weight conversion,
  native Q6 head or batched-decode changes are imported.
- Areas: Flash-Next expert dispatch, routed F16 kernels and device codebook;
  independent F64 operator checks with IQ and short/ragged batches.
- Validation: Windows GPU build and expanded `routed_wmma_ops` pass (83.26 s),
  including 24 IQ cases under unchanged limits. All 23 compiled routed
  variants have zero scratch/register spills; all 512 device IQ3_S grid
  entries match the independent CPU decoder. Formatting, documentation,
  dependency checks and ten resource-parser tests pass. Full-model checks
  remain in progress; Linux is unrun.
  [Criteria and evidence](docs/plans/iq-experts-prefill.md).
- Status: experimental candidate; intentional IQ prefill arithmetic change
  is not yet qualified or promoted.

### 2026-10-17 — Keep the cached conversation when tool choice changes

- Source: [gufo-org/gufo@23d970d0ca0ef694dafec64131269f9c24590940](https://github.com/gufo-org/gufo/commit/23d970d0ca0ef694dafec64131269f9c24590940)
  (PR #623), adapted.
- Local: commit introducing this entry.
- Adaptation: same goal, different route. Upstream deletes the `required`
  instruction from the system prompt because its grammar forces a tool call.
  This fork enforces `required` only after generation
  (`tool_choice_unsatisfied`), so the instruction is kept and moved after the
  cacheable prefix instead. Qwen renders it as a trailing user turn after
  `stable_prefix_bytes` (system turns must lead), only with tools and a
  generation prompt. DeepSeek appends it after the last message, before the
  generation prompt; its continuation-render frontier already stops before
  it. `require_tool_call` options and the server wiring are unchanged.
  Upstream's functional suite is omitted (absent here). Cost: the request after
  a `required` one resumes from the last-message checkpoint rather than the
  live frontier, re-processing that assistant reply.
- Areas: `src/models/qwen/chat_template.cpp`,
  `src/models/deepseek_v4_flash/chat_template.cpp`, `docs/SERVER.md`,
  `tests/models/qwen/tokenization/chat_template_test.cpp`,
  `tests/models/deepseek_v4_flash/chat_template_test.cpp`.
- Validation: Windows `gpu-test` build; `qwen_chat_template_test` (required
  and auto share the stable prefix; the turn follows it; absent without a
  generation prompt) and `ds4_chat_template_test` (required only appends
  after the last message; pinned golden unchanged) pass and fail against the
  previous renderers. `openai_chat_test` and `text_model_runner_test` pass.
  Real-model check on Flash-Next UD-IQ4_XS, `gufo serve -c 16384`,
  temperature 0, thinking off, three tools, 20 prompts that need no tool, all
  with `tool_choice: "required"`: the previous system-prompt placement gave
  18 `tool_choice_unsatisfied`, 1 call and 1 length-limited text reply; the
  trailing turn gave 19 calls and 1 unsatisfied. Upstream's deletion renders
  `required` like `auto`, where all 20 replied with text (20 unsatisfied).
  Switching a ~1K-token conversation from `auto` to `required` reused 0
  tokens before (1,013 prefilled) and 989 after (28 prefilled). DeepSeek not
  run on a real model (no local weights). Linux CI not run.
- Status: retained.

### 2026-10-17 — Compact Flash-Next sparse attention selections once per row

- Source: [gufo-org/gufo@baf5a1bbab5b99a8278c4869122fd173b205671b](https://github.com/gufo-org/gufo/commit/baf5a1bbab5b99a8278c4869122fd173b205671b)
  (PR #622).
- Local: commit introducing this entry; builds on `bebff14c` (PR #292).
- Adaptation: imported the per-row selection compaction: for narrow split
  batches (at most 8 rows) at ratio 4, each row's selected blocks are written
  once to `attn_lists` (row × `indexer_top_k / compress_ratio` entries) with a
  count in `attn_counts`, and every split reads that list instead of
  re-scanning the mask. Null lists keep the mask walk, which wide batches still
  use. The executor call conflicted with upstream's #518 `SigmoidMulQ8_1`
  epilogue, which this fork does not have; the list arguments were added to
  the existing `Attention` + `SigmoidMul` sequence. Upstream's benchmark
  artifacts are omitted.
- Areas: `src/models/qwen38_flash_next/kernels/rocm/{kernels.hip.cpp,kernels.hpp,executor.cpp,executor.hpp}`,
  `tests/models/qwen38_flash_next/attention_ops_test.cpp`.
- Validation: Windows `gpu-test` build; `qwen38_flash_next.attention_ops`
  passes, and the list path is bitwise equal to the mask walk for 1, 3 and 8
  rows at starts 1,500, 8,192, 32,768 and 163,840. `gufo bench --logit-eval`
  on UD-IQ4_XS with the Q8_0 MTP sidecar, same 64 KB corpus and `2,4,8:p8`
  schedules as the #292 entry: all 15,834 raw-logit rows match `bebff14c`
  exactly in both schedules. AR `gufo bench -p 512 -n 128 -d 0,32768,131072
  -r 2`, same session as the #292 timing: decode 28.31 tok/s at depth 0
  (28.45 for #292), 27.50 at 32K (27.34, +0.6%) and 26.91 at 128K (26.60,
  +1.2%); against `daf56a19` that is +4.4% at 32K and +9.5% at 128K. pp512
  unchanged within noise. Linux CI not run.
- Status: retained.

### 2026-10-17 — Cut Flash-Next sparse attention tiles across selection windows

- Source: [gufo-org/gufo@6a4c89784adb1c99830889b42fc3e243c3fcb7d8](https://github.com/gufo-org/gufo/commit/6a4c89784adb1c99830889b42fc3e243c3fcb7d8)
  (PR #292).
- Local: `bebff14c63b2c08ed2317a0a07dcf73c1a61a313`; snapshot compatibility
  correction in the commit updating this entry.
- Adaptation: imported the `AttentionKernel` change that cuts key tiles from
  the running compacted block list, carrying a partial tile of at most
  `kTile / 4` blocks into the next 1024-block window, and deals the tiles
  round-robin across splits. Previously each window was tiled separately, so
  deep contexts produced many mostly empty tiles and windows with less than a
  tile landed on split 0. The selected keys are unchanged; only tile grouping
  and softmax accumulation order differ. Also imported the FP64 structured
  attention tests. Upstream's `gpu_probe` additions, model-identity record and
  benchmark artifacts are omitted (Linux-specific harness data).
  The local snapshot compatibility version is 20 because accumulation order
  changes: version-19 and earlier checkpoints are rebuilt from the prompt.
  Older builds cannot read version-20 checkpoints after a rollback.
- Areas: `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`,
  `src/models/qwen38_flash_next/engine.hpp`,
  `tests/models/qwen38_flash_next/attention_ops_test.cpp`,
  `tests/models/qwen38_flash_next/snapshot_test.cpp`, `docs/SERVER.md`.
- Validation: Windows `gpu-test` build; `qwen38_flash_next.attention_ops`
  passes with FP64 worst absolute errors ≤ ~3e-7. Arithmetic changes
  intentionally, so the declared limit is drift below the existing
  path-to-path spread. `gufo bench --logit-eval` on UD-IQ4_XS with the Q8_0
  MTP sidecar compared `daf56a19` against this commit. The corpus was the
  first 64,000 bytes of wikitext-2 test (sha256 `b12cec83c3206e1b…`, 15,834
  positions; the 16 KB corpus stays inside one window and cannot exercise the
  change), schedules `2,4,8:p8`. `p8` rows match exactly (perplexity
  2.000619). Narrow verify batches (`2,4,8`) first differ at position 4,105:
  top-1 agreement 97.22%, mean KL over the top 64 tokens 1.50e-2, perplexity
  1.973978 → 1.976805. Both builds are equally distant from the `p8` path
  (top-1 94.66%, mean KL 4.49e-2 before and 4.54e-2 after), so the drift is a
  third of the existing schedule spread. AR `gufo bench -p 512 -n 128
  -d 0,32768,131072 -r 2` on UD-IQ4_XS, base run before and after: decode
  28.33/28.43 → 28.45 tok/s at depth 0, 26.38/26.34 → 27.34 at 32K (+3.7%),
  24.58/24.57 → 26.60 at 128K (+8.2%); pp512 unchanged within noise. Linux CI
  not run.
- Snapshot compatibility follow-up: native Windows/gfx1151, TheRock 10.0.0
  (clang 23), `gpu-test`; `gufo` and the snapshot test build. Running
  `qwen38_flash_next_snapshot_test --model "$MODEL" --mtp-model "$MTP"`
  on UD-IQ4_XS with the shared Q8_0 MTP sidecar (8192 context, seed 7,
  temperature 0.8, top-k 40, top-p 0.9) fails before the bump with
  `incompatible snapshot version accepted`, then passes at version 20.
  AR/MTP incremental captures, persistent round trips, exact restored logits,
  sampled replay and deferred-residual RNG replay pass. The disk-cache test,
  repository formatting and docs checks pass. This follow-up changes metadata
  only; no new logit/perplexity measurements or Linux execution.
- Status: retained.

### 2026-10-17 — Flat-table Flash-Next n-gram gather

- Source: [gufo-org/gufo@8497711eb154ecff55ef4119a01ace214149e276](https://github.com/gufo-org/gufo/commit/8497711eb154ecff55ef4119a01ace214149e276)
  (PR #613), gather half only.
- Local: commit introducing this entry.
- Adaptation: imported `NgramTable::ReserveGather` and the flat power-of-two
  deduplication table that replaces the per-gather `std::unordered_map`, plus
  the `ReserveGather(T * ple_heads)` call in `Executor::Create`. Upstream's
  prompt-lookup MTP proposals, length-controller changes, snapshot payload bump
  and functional suite are omitted. They conflict with this fork's own
  `prompt_lookup.hpp` and remain a separate decision. The fork's cached
  overlapped reads are unchanged. Scratch is reserved once at load:
  4,096 prefill tokens × 16 heads gives 65,536 rows, a 1 MiB table plus about
  2 MiB of job/copy lists, independent of context length.
- Areas: `src/models/qwen38_flash_next/ngram.{hpp,cpp}`,
  `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`,
  `tests/models/qwen38_flash_next/ngram_test.cpp`.
- Validation: Windows `gpu-test` build; `qwen38_flash_next.ngram` passes,
  including upstream's reuse and active-gather cases. `gufo bench --logit-eval`
  on Flash-Next UD-Q4_K_XL with Q8_0 MTP compared the previous commit against
  this one, same toolchain and settings. The corpus was the first 16,000 bytes
  of wikitext-2 test (sha256 `fdcdcd05f34de43e…`), with schedules
  `1:2,4,8:p8`. All 3,686 raw-logit rows match exactly in every schedule
  (perplexity 1.443348 / 1.443348 / 1.472327). Repeated on the fork's UD-IQ4_XS
  qualification weights with the same Q8_0 MTP sidecar and corpus: all 3,686
  rows match exactly (perplexity 1.544504 / 1.544504 / 1.576038). No throughput
  measurement. Linux CI not run.
- Status: retained.

### 2026-10-17 — Prepare the Qwen vision sidecar before serving

- Source: [gufo-org/gufo@0604e9ebc41388ab610ed0bed198e49ef96a6a82](https://github.com/gufo-org/gufo/commit/0604e9ebc41388ab610ed0bed198e49ef96a6a82)
  (PR #616).
- Local: commit introducing this entry.
- Adaptation: imported as is (public `Encoder::Upload`, `PrepareVision` at both
  Qwen `InferenceBackend::load` overloads, docs and test), with two Windows
  changes. The `event=vision_prepared` free-memory field and the new test read
  `gufo::platform::DeviceMemoryInfo` instead of `hipMemGetInfo`, because
  `hipMemGetInfo` reports wrong numbers on Windows. On Windows the first encode
  keeps a fixed ~73 MiB of one-time runtime state, beyond upstream's 64 MiB
  allowance. The test now bounds the first encode by half the sidecar, which
  still catches a re-upload, and requires a repeated encode to stay flat.
- Areas: `src/models/qwen/vision/encoder.{hpp,hip}`,
  `src/cli/serve/inference_backend.cpp`, `docs/SERVER.md`,
  `tests/models/qwen/vision_prepare_test.cpp`, `tests/models/qwen27b/CMakeLists.txt`.
- Validation: Windows `gpu-test` build. `qwen_vision_prepare_test` passes on
  `mmproj-BF16.gguf` (resident 903,984,064 bytes, matching the sidecar
  weights); upstream's unmodified check fails there as described.
  `qwen_vision_discovery_test` passes; `qwen_vision_serving_test` skipped (no
  model argument under CTest). Real serve smoke test on Flash-Next UD-Q4_K_XL
  with auto-discovered BF16 projector, two runs per build against a
  `6d7f8d3d` release baseline: `vision_prepared` logs identity ~1.11 s and
  upload ~0.17 s. Readiness takes +1.0 s. The first image request drops from
  2.72/2.54 s to 0.92/0.89 s (queue 1.1–1.2 s to under 10 ms) with identical
  replies. Second-image latency is unchanged, and `/v1/models` and `/props`
  still report image input. On UD-IQ4_XS with the same auto-discovered BF16
  projector, readiness is 21.8 s, `vision_prepared` logs identity 1.62 s and
  upload 0.18 s, and the first image request takes 1.11 s (queue 6 ms) with
  image input advertised. That run is candidate-only, with no IQ4_XS baseline.
  Linux CI not run.
- Status: retained.

### 2026-10-17 — Decimal spelling of integer JSON numbers

- Source: [gufo-org/gufo@22e36769f9b25a57963a9215003eae0f5f9c15d6](https://github.com/gufo-org/gufo/commit/22e36769f9b25a57963a9215003eae0f5f9c15d6)
  (PR #614).
- Local: commit introducing this entry.
- Adaptation: `json.hpp` imported unchanged, so integers up to 2^53 serialize
  in decimal (`100000`, not `1e+05`). This affected streamed tool arguments
  and Qwen typed tool-call replay (`json::parse(...).tojson()`). Upstream's
  `json_constraint_test` and functional-suite edits are omitted because those
  files do not exist here. Its Responses replay case targets a test absent
  here and is replaced by an integer case in `TestTypedToolReplay`.
- Areas: `src/core/json.hpp`, `tests/core/json_test.cpp`,
  `tests/cli/openai_chat_test.cpp`,
  `tests/models/qwen/tokenization/chat_template_test.cpp`.
- Validation: Windows `gpu-test` builds of `json_test`, `openai_chat_test` and
  `qwen_chat_template_test` pass. With the previous `json.hpp`, `json_test` and
  `qwen_chat_template_test` both fail. Repository format check passes. There are
  no numerics changes, so no logit checks were run. Linux CI not yet run.
- Status: retained.

### 2026-10-15 — Bound Flash-Next selector score scratch

- Source: [gufo-org/gufo@5a0a0ed75009ad56f9eee7cdd92f3b1752dfee64](https://github.com/gufo-org/gufo/commit/5a0a0ed75009ad56f9eee7cdd92f3b1752dfee64)
  (PR #463), following the eager-grid adaptation below.
- Local: `250b7a2194af146bc2dd8fdf952d55eeab658fc6`.
- Adaptation: cap shared scores at 20 MiB for the model's 262K context while
  retaining captured decode capacity. Compact eager score strides to populated
  context and reuse the allocation across bounded selector chunks. Preserve the
  fork's MTP mask range; omit final normalization/output pruning from #463.
- Areas: Flash-Next executor allocation/selection and a cleared-output,
  exactly-sized compact-score regression at deep and ragged positions.
- Validation: fresh Windows release/GPU builds and full/bounded/compact mask
  equality pass, retaining independent F64 scores and tie ordering. Formatting,
  original resource guard and twelve scratch/spill-free dense F16 variants
  pass. Final matched-model/state checks pass; measured throughput remains
  inconclusive. See the
  [series record](docs/plans/prefill-small-pr421-571-463.md).
- Status: retained in the isolated worktree; Windows numerical/state checks
  pass. No native speedup is established; Linux CI remains unrun.

### 2026-10-15 — Wider Flash-Next F16 router tiles

- Source: [gufo-org/gufo@e8a75023f13fdfb8d77ff39efdf0da3b2c6f0fd5](https://github.com/gufo-org/gufo/commit/e8a75023f13fdfb8d77ff39efdf0da3b2c6f0fd5)
  (PR #571).
- Local: `7a564c32a374a68f836be2198b0f27c0756e126b`.
- Adaptation: use the existing GEMM template's 128-by-128 tile for unquantized
  F16 projections of at least 1024 tokens. Keep its two K16 chains and the
  original narrow route. No other #571 kernels or fusions are imported.
- Areas: Flash-Next `kernels.hip.cpp` unquantized projection dispatch.
- Validation: fresh Windows release/GPU builds, projection/F16 replay/GDN
  operators pass (3/3), including exact wide/narrow split output and independent
  F64 checks. Formatting, original resource guard and twelve scratch/spill-free
  dense F16 variants pass. Final model/state checks pass; measured throughput
  remains inconclusive. See the
  [series record](docs/plans/prefill-small-pr421-571-463.md).
- Status: retained in the isolated worktree; Windows numerical/state checks
  pass. No native speedup is established; Linux CI remains unrun.

### 2026-10-15 — Smaller Flash-Next attention projection stage

- Source: [gufo-org/gufo@9e0426620ba1c7f02aafac1b846276f2f9b9e361](https://github.com/gufo-org/gufo/commit/9e0426620ba1c7f02aafac1b846276f2f9b9e361)
  (PR #421), following the row-tile ordering adaptation below.
- Local: `dec9932e92d5cb92cea2d0c8fd287a58d2cf12ac`.
- Adaptation: stage one K block in the fused attention projection, reducing
  LDS from 48 to 24 KiB without changing the K16 accumulation order. Keep SSM
  and output projection staging unchanged.
- Areas: Flash-Next `kernels.hip.cpp` fused attention template and dispatch.
- Validation: fresh Windows release/GPU builds, projection and attention
  operators pass (2/2), including ragged fused Q/K/V outputs. Formatting,
  original resource guard and eleven scratch/spill-free dense F16 variants
  pass. Final matched-model/state checks pass; measured throughput remains
  inconclusive. See the
  [series record](docs/plans/prefill-small-pr421-571-463.md).
- Status: retained in the isolated worktree; Windows numerical/state checks
  pass. No native speedup is established; Linux CI remains unrun.

### 2026-10-15 — Group Flash-Next projection row tiles

- Source: [gufo-org/gufo@0690047f6cd2bd86b7536c43e9c7fa353bddc2a2](https://github.com/gufo-org/gufo/commit/0690047f6cd2bd86b7536c43e9c7fa353bddc2a2)
  (PR #421).
- Local: `73fa2bbfe673e78609586b349af22ccfd1ec3f08`.
- Adaptation: reuse the existing row-group mapping for wide output, attention,
  shared-expert down and fused SSM projections. Only launch order changes;
  retain each output's K accumulation and all narrow/fallback routes.
- Areas: Flash-Next `kernels.hip.cpp` dispatch and fused-epilogue guards.
- Validation: fresh Windows release/GPU builds, projection/attention/GDN and
  F16 library replay operators pass (4/4); formatting and resource checks pass,
  including eleven zero-scratch/spill dense F16 variants. Final matched-model
  and state checks pass; measured throughput remains inconclusive. See the
  [series record](docs/plans/prefill-small-pr421-571-463.md).
- Status: retained in the isolated worktree; Windows numerical/state checks
  pass. No native speedup is established; Linux CI remains unrun.

### 2026-10-15 — Bound eager Flash-Next selector grids

- Source: [gufo-org/gufo@109a577da5feaffe669e88a6baa9c45e35c3a815](https://github.com/gufo-org/gufo/commit/109a577da5feaffe669e88a6baa9c45e35c3a815)
  (PR #421).
- Local: `2f9a071ea7b843ed9e3aaefd0adfdf01657a8210`.
- Adaptation: bound eager prefill score grids by the complete live blocks;
  captured decode keeps its maximum grid and score stride. Preserve the fork's
  MTP catch-up mask range. Omit the other #421 kernels from this improvement.
- Areas: Flash-Next executor and selection kernel/interface; cleared-output
  selection regression and a 32K extension of the existing logit diagnostic.
- Validation: fresh Windows release/GPU builds, bounded/full-grid operator
  equality and repository formatting pass. The existing resource contract and
  nine dense F16 variants remain scratch/spill-free. Final matched-model
  logits/perplexity and state checks pass; measured throughput remains
  inconclusive. See the
  [series record](docs/plans/prefill-small-pr421-571-463.md).
- Status: retained in the isolated worktree; Windows numerical/state checks
  pass. No native speedup is established; Linux CI remains unrun.

### 2026-10-13 — Reduce-scatter for narrow batched Q8 decode

- Source: [gufo-org/gufo@c1f01311318e6c677fd25c268b51a4e94ca324cc](https://github.com/gufo-org/gufo/commit/c1f01311318e6c677fd25c268b51a4e94ca324cc)
  (PR #518), following the recorded DPP transport and few-row read-ahead.
- Local: 76656a5befc12b2a9171895654c24c4a2524a42a.
- Adaptation: distribute the same descending 32-lane butterfly sums across
  lanes for two through eight Q8 inputs, including gated projections. Keep
  scalar dispatch, wide/ragged paths and separate wave64 expert routes intact.
- Areas: Flash-Next `mmq/common.hpp` and `mmq/mmvq.hip.cpp`.
- Validation: fresh Windows builds and the projection operator pass with strict
  scalar/batched equality at widths 1–8 and wide/ragged controls. Both formats
  retain all 696 complete matched target rows and exact perplexity. The compiled
  prefill resource guard passes 57 kernels; 83 relevant decode variants have
  zero scratch/spills. Final prefill/native checks pass; functional timing and
  Linux remain unqualified; see the
  [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-13 — Read ahead for few-row Q8 decode

- Source: [gufo-org/gufo@0aa6f75af25639504a6aaaa14946d6f47f31f3dd](https://github.com/gufo-org/gufo/commit/0aa6f75af25639504a6aaaa14946d6f47f31f3dd)
  (PR #518).
- Local: b713fe95db8913496c0cb6e512252b9293e6c31f.
- Adaptation: load twenty lane-local Q8 blocks ahead for complete long rows
  with one input and at most 64 outputs. Preserve scalar products, FMA order,
  original reduction and all other dispatch paths. No producer fusion.
- Areas: Flash-Next `mmq/mmvq.hip.cpp` and projection operator fixtures.
- Validation: fresh Windows release/operator builds and the expanded projection
  test pass, including 4×10240 scalar-versus-batched outputs. All 696 complete
  matched target logit rows and six perplexities remain exact on both formats.
  Final prefill/native checks pass; functional timing and Linux remain
  unqualified; see the [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-13 — Cached Flash-Next norms and MTP row reads

- Source: [gufo-org/gufo@e1b20ba9e11540e817116ae196b51696cc80d71a](https://github.com/gufo-org/gufo/commit/e1b20ba9e11540e817116ae196b51696cc80d71a)
  (PR #518), using the previously recorded materialization helper.
- Local: e480c3af62a6b4de7221f18d5d665e067607cbbd.
- Adaptation: retain 2560/10240-element norm inputs/gamma in registers with
  explicit original square/FMA order, and vectorize aligned MTP copy/add rows.
  General geometry/alignment fallbacks and full-width normalization remain.
- Areas: Flash-Next `kernels.hip.cpp`.
- Validation: fresh Windows builds, three affected GPU operators and the
  57-kernel prefill resource contract pass. All 696 matched target logit rows
  and perplexities are exact. A new explicit diagnostic built against the
  original `ab4d98189a6d` kernels captures eight fixed MTP states per format;
  both formats retain all 128 stage hashes, including 16 complete vocabulary
  rows, with finite values and exact isolated batch/catch-up comparisons.
  The original IQ4_XS independent scalar MTP oracle fails attention relative RMS
  0.0093689 against 0.002; its limits remain unchanged. The original UD-Q4_K_XL
  independent oracle passes. This is regression parity, not new IQ4 oracle
  qualification. Final prefill/native checks pass; functional timing and Linux
  remain unqualified; see the [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-13 — Vector reads for small Flash-Next F32 projections

- Source: [gufo-org/gufo@482fd027d109c160dc97f307202f717c05fc5bc5](https://github.com/gufo-org/gufo/commit/482fd027d109c160dc97f307202f717c05fc5bc5)
  (PR #518), following the recorded small-kernel load scheduling adaptation.
- Local: 0f5900d94dc44a3433821e016201175402e8ee8e.
- Adaptation: use 16-byte reads for aligned, whole-pass F32 rows, with explicit
  original FMA order. Preserve the generic path for misaligned, ragged and
  non-F32 inputs, including grouped projection dispatch.
- Areas: Flash-Next `kernels.hip.cpp` and projection operator fixtures.
- Validation: fresh Windows production/operator builds and the projection
  operator test pass, including deliberate four-byte misalignment, FP64 formulas,
  allocation-boundary tails, every narrow width and grouped widths through 64.
  All 696 matched full-logit rows and perplexities remain exact across the two
  target quantizations. The 57-kernel prefill resource contract passes. Final
  prefill/native checks pass; functional timing and Linux remain unqualified;
  see the [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-13 — Flash-Next decode load scheduling

- Source: [gufo-org/gufo@6e52e0f8a30fa609b8e2a0a817c2e48b2875a963](https://github.com/gufo-org/gufo/commit/6e52e0f8a30fa609b8e2a0a817c2e48b2875a963),
  [19cbf32fabea093c161fc018f9c36a0bfba8db0a](https://github.com/gufo-org/gufo/commit/19cbf32fabea093c161fc018f9c36a0bfba8db0a),
  [773fcfb3147876c9cc52d8d53954f57c142d5544](https://github.com/gufo-org/gufo/commit/773fcfb3147876c9cc52d8d53954f57c142d5544),
  and only the materialization helper from
  [11827a9a809cab06ab447054b56c122770b3f25b](https://github.com/gufo-org/gufo/commit/11827a9a809cab06ab447054b56c122770b3f25b)
  (PR #518).
- Local: 036dc3243e5aa5246531eb9bf6d7f30c5a96f794.
- Adaptation: preload bounded small-GEMM, HC and MoE operands while keeping
  the original product/add order and existing output layouts. Specialize the
  residual register cache to complete 2560-element rows; ragged shapes keep
  guarded reads instead of clamped unused residual reads. Preserve null-gamma
  handling and the Windows HC-down/SiLU route. No producer Q8_1 fusion.
- Areas: Flash-Next `kernels.hip.cpp` and projection operator fixtures.
- Validation: fresh Windows production/operator builds and four affected GPU
  tests pass, including independent projection/HC formulas and allocation-ended
  ragged F32 rows. Both quantizations retain all 696 matched full-logit rows and
  exact perplexity. All 57 prefill resource-contract kernels pass; 32 inspected
  changed-family variants have zero scratch/spills. Final prefill/native checks
  pass; functional timing and Linux remain unqualified; see the
  [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-13 — Branch-free Flash-Next Q4 scale reads

- Source: [gufo-org/gufo@a17f9ea148de29a6b2b95afed89a5411c530f799](https://github.com/gufo-org/gufo/commit/a17f9ea148de29a6b2b95afed89a5411c530f799)
  and [gufo-org/gufo@d1a8a8e26311b366a8cfa374af9db59cf1e3be3e](https://github.com/gufo-org/gufo/commit/d1a8a8e26311b366a8cfa374af9db59cf1e3be3e)
  (PR #518).
- Local: d4099648739d12fa6e73bf25c2580b96748bca11.
- Adaptation: reuse the final combined integer scale helper for scalar and
  grouped Q4_K dots, including existing wave64 callers. Keep the fork's native
  IQ expert paths, tensor encodings and projection dispatch. No routing changes.
- Areas: Flash-Next `mmq/vecdotq.hpp`.
- Validation: fresh Windows production/operator builds pass; projection and
  routed-WMMA operator tests pass. All 696 matched complete logit rows and
  per-schedule perplexities match the `ab4d98189a6d` baseline across IQ4_XS and
  UD-Q4_K_XL. A separate CPU algebra check covers all 256 byte values in each
  scale-byte position plus 5,000 random packings (8,072 cases × four sub-blocks);
  this supplements the compiled-kernel checks. Final prefill/native checks pass;
  functional timing and Linux remain unqualified; see the
  [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-13 — Flash-Next DPP wave transport

- Source: [gufo-org/gufo@28b3c03b43f6951a5d13c2031cf1b6ced1ae6b52](https://github.com/gufo-org/gufo/commit/28b3c03b43f6951a5d13c2031cf1b6ced1ae6b52)
  (PR #518).
- Local: 3fdb31ec5a6d5a3bfc50da7bbc89d11b6d61e270.
- Adaptation: replace remaining XOR shuffle transports with gfx1151 DPP and
  permlanex16, retaining the existing reduction loops, operands and fallback.
  Keep the fork's ascending GDN DPP tree and its wave64 expert dispatch with
  logical 32-lane reductions. Producer Q8_1 fusion is not imported.
- Areas: Flash-Next `kernels.hip.cpp` and `mmq/common.hpp`.
- Validation: fresh TheRock 10.0.0 Windows production/operator builds pass;
  all 15 Flash-Next CPU/operator tests pass. Against the preserved `ab4d98189a6d`
  baseline, IQ4_XS and UD-Q4_K_XL each retain all 348 complete vocabulary rows
  across scalar decode, verification widths 1–8 and eight-row prefill, with
  exact perplexity. All 57 prefill resource-contract kernels pass; the 38
  inspected decode/attention variants retain zero scratch and spills.
  Final prefill/native checks pass; functional timing and Linux remain
  unqualified. See the
  [implementation record](docs/plans/decode-kernels-pr518-stages01.md).
- Status: held in the isolated implementation worktree. Windows numerical/state
  gates pass and decode gains are measured; full functional-latency and Linux
  qualification remain outstanding. Final evidence and limits are recorded in
  [the implementation record](docs/plans/decode-kernels-pr518-stages01.md).

### 2026-10-12 — Completion-route prompt cache controls

- Source: [gufo-org/gufo@adc3f0f59b32d1e6ceadcb7b2664f38e24a40cd3](https://github.com/gufo-org/gufo/commit/adc3f0f59b32d1e6ceadcb7b2664f38e24a40cd3)
  and [gufo-org/gufo@4528634fe2474cc1f9099cbdd2e81e823054aee2](https://github.com/gufo-org/gufo/commit/4528634fe2474cc1f9099cbdd2e81e823054aee2)
  ([PR #557](https://github.com/gufo-org/gufo/pull/557)).
- Local: commit introducing this entry; based on
  `11f83fa563b502e3788791e911bd0f7a624bc10e`.
- Adaptation: forward `cache_prompt` on both raw completion routes through the
  fork's synchronous `complete()` interface to the existing scheduler. Also
  close the fork's Responses forwarding gap, which upstream had already fixed.
  All four routes honor booleans, retain the `true` default for omitted or null
  values, and reject other types before generation. Preserve cache publication
  after a bypassed request. Omit upstream's newer raw streaming/deferred-request
  interface changes because this fork does not have that interface; no scheduler,
  cache format, kernel or toolchain changes are needed.
- Areas: `src/cli/serve/http_server.cpp`,
  `src/cli/serve/inference_backend.cpp`, `src/cli/serve/inference_backend.hpp`,
  `src/cli/serve/text_generation_backend.hpp`, `src/cli/serve/openai_chat.cpp`,
  `tests/cli/http_server_test.cpp`, `tests/cli/openai_chat_test.cpp`,
  `docs/SERVER.md`, `docs/KV-CACHE.md`.
- Validation: fresh Windows CPU builds with TheRock 10.0.0 (clang 23,
  MSVC 14.51) pass `http_server_test`, `openai_chat_test`,
  `text_model_runner_test` and `text_generation_scheduler_test`. HTTP fixtures
  cover all four routes, sequential false-to-default requests, null, invalid
  types without backend calls, and buffered/streaming Chat and Responses.
  Existing runner fixtures cover disk/live/RAM bypass and later cache publication.
  The Windows HIP production build passes. An isolated gfx1151 HTTP smoke with
  Flash-Next IQ4_XS, Q8_0 MTP (draft limit 7), BF16 projector, context 4096,
  one session, temperature 0 and seed 123 reproduces the baseline bug and passes
  36 candidate requests: bypass has zero cached tokens, true/omitted/null reuse
  the cache, and invalid strings return 400. Buffered timing counters confirm
  bypass prefills the full prompt; streaming Chat/Responses forwarding is covered.
  Against baseline `11f83fa563b5`, the matched 24-token fixed-prose corpus has
  exact SHA-256 equality for all 22 complete raw-logit rows in each of three
  schedules (decode 1, mixed verification 1/2/4/8, prefill 8). Perplexities match
  exactly at 6.238422, 6.238422 and 7.232463, respectively. This is a bounded
  arithmetic regression check, not a full model-quality qualification.
  Formatting (clang-format 21.1.8), documentation and diff whitespace checks pass.
  Commands, build/source identities, artifact/corpus hashes, requests/responses
  and logs are retained under ignored `build/pr557/` in the implementation
  worktree. Linux execution, GPU restart/disk-cache smoke, full quality suites
  and performance comparisons remain unrun. Test-owned GPU processes are stopped.
- Status: retained.

### 2026-10-10 — Decode narrow integer GGUF metadata arrays

- Source: [gufo-org/gufo@3fac1bb51c97ef3606bf0bc662bf778344414cb4](https://github.com/gufo-org/gufo/commit/3fac1bb51c97ef3606bf0bc662bf778344414cb4)
  ([PR #532](https://github.com/gufo-org/gufo/pull/532), reviewed head
  [4c4df642b00eae60e276b2d58ce2836c3dc48373](https://github.com/gufo-org/gufo/commit/4c4df642b00eae60e276b2d58ce2836c3dc48373)).
- Local: commit introducing this entry; based on
  `e2454792a37934bf19bf2d84ae085553ee8f73e4`.
- Adaptation: import the parser and regression fixture unchanged; retain the
  fork's Windows temporary-path and mapped-file truncation test adaptations.
  Widen unsigned/signed 8- and 16-bit arrays to the existing 64-bit vectors,
  preserving signs and rejecting scalar access to arrays. Existing bounds and
  the 128 MiB metadata budget remain unchanged. No prerequisites or omissions.
- Areas: `src/core/gguf_reader.cpp`, `tests/core/gguf_reader_test.cpp`.
- Validation: fresh Windows `cpu-test` builds with TheRock 10.0.0 (clang 23,
  MSVC 14.51) pass `gguf_reader_test`, `gguf_identity_test` and
  `qwen38_flash_next.config`. The upstream regression fixture fails with
  `bad variant access` when built against the baseline parser, then passes after
  restoring the fix. An isolated CPU metadata/configuration probe opens all
  three local Flash-Next UD-IQ4_XS shards and the Q8_0 MTP sidecar successfully;
  descriptor counts, selected geometry and RoPE/compression/PLE array values
  match the baseline. This is metadata loading, not GPU weight upload or inference.
  Repository formatting (clang-format 21.1.8), documentation and diff whitespace
  checks pass. Commands and logs are retained under ignored `build/pr532/` in
  the implementation worktree. GPU model-load smoke, matched-token full logits,
  perplexity, sanitizers and Linux execution remain unrun. DS4's loader test is
  HIP-only and was not built in this CPU qualification. No performance claim.
- Status: retained.

### 2026-10-09 — Tool-result image parser compatibility

- Source: [gufo-org/gufo@a5df2744dbd97c8ff60da25a1ab9daf90fffc639](https://github.com/gufo-org/gufo/commit/a5df2744dbd97c8ff60da25a1ab9daf90fffc639)
  ([PR #506](https://github.com/gufo-org/gufo/pull/506)).
- Local: commit introducing this entry.
- Adaptation: allow Chat tool-message image parts through the existing image
  parser; accept Responses custom-tool calls/results as stateless history,
  decoding free-form input as one literal string argument named `input`.
  Retain earlier-call matching, image budgets and role/detail/URL validation.
  Reuse existing Responses function-result images and Qwen tool-image rendering,
  including literal text and outer whitespace preservation. Omit upstream's
  renderer replacement and functional/Pi harness additions: the renderer already
  supports the required behavior and focused native fixtures cover the parser
  changes. Custom-tool generation remains unsupported.
- Areas: `src/cli/serve/openai_chat.cpp`, focused API/HTTP fixtures and
  [server contracts](docs/SERVER.md).
- Validation: fresh Windows CPU builds with TheRock 10.0.0 pass
  `openai_chat_test`, `http_server_test` and `qwen_chat_template_test`, with
  assertions enabled. Fixtures cover mixed/image-only observations, image order,
  literal custom input, call identity, malformed/unmatched history, role/image
  restrictions, rejection recovery and buffered/streamed forwarding. Repository
  C++ formatting (clang-format 21.1.8), documentation and diff whitespace checks
  pass. Real-model/GPU smoke, matched-token logits/perplexity and Linux execution
  unrun; this adaptation changes transport parsing only, with no renderer,
  inference arithmetic or cache changes. No performance claim.
- Status: retained.

### 2026-10-09 — MTP catch-up attention boundary

- Source: [gufo-org/gufo@fd747a51951ccd09eda20de5c34313670dc9b9d3](https://github.com/gufo-org/gufo/commit/fd747a51951ccd09eda20de5c34313670dc9b9d3)
  ([PR #485](https://github.com/gufo-org/gufo/pull/485), reviewed head
  [1a2b49476d140146831b68f450ceac76cb0e8618](https://github.com/gufo-org/gufo/commit/1a2b49476d140146831b68f450ceac76cb0e8618)).
- Local: commit introducing this entry.
- Adaptation: extend the fork's existing dense/sparse attention split to
  last-only MTP catch-up and score all masks consumed by its shifted final
  sparse group. Preserve the fixed 512-query selector chunk, projections,
  arithmetic, refusal/fallback and Windows tuning. Bump snapshots to version 17
  to reject previously computed predictor state. Omit upstream checkpoint
  machinery and every other optimization in the PR. Extract the existing
  full/tail predictor comparison into an explicit test-only audit entry point
  and extend it through all sparse query-group alignments; retain the independent
  scalar oracle and its unchanged tolerances.
- Areas: Flash-Next executor and snapshot compatibility, GPU probe/catch-up
  audit, cache documentation and the
  [qualification record](docs/mtp-boundary-pr485-validation.md).
- Validation: the baseline reproduces a 224-row catch-up failure at position
  2025. Fresh pinned Windows builds pass the focused full/tail oracle over
  seven widths and 55 rounds, three GPU operator targets, the 57-kernel resource
  guard and formatting. Isolated selector scratch also detects an old-mask-only
  negative control at position 1806 with 257 rows. All 270 full-logit rows and
  corpus perplexity match baseline exactly; session, snapshot, rollback and BF16
  image-prefix suites pass. The independent scalar MTP oracle fails identically
  before catch-up on baseline and candidate; this is not counted as a pass.
  Details are in the qualification record. Linux CI and other quantizations
  are unrun.
- Status: retained; extends the PR #470 attention adaptation below.

### 2026-10-08 — Configurable retained RAM snapshot cap

- Source: configurable-cap outcome from
  [gufo-org/gufo@d7cf7ee4b8e3a506ee1ec9c301e8a3efd10c6c94](https://github.com/gufo-org/gufo/commit/d7cf7ee4b8e3a506ee1ec9c301e8a3efd10c6c94)
  ([PR #369](https://github.com/gufo-org/gufo/pull/369)); ceiling behavior reviewed in
  [gufo-org/gufo@53c590649295edf63abcc06117231cdc890b69c7](https://github.com/gufo-org/gufo/commit/53c590649295edf63abcc06117231cdc890b69c7)
  ([PR #384](https://github.com/gufo-org/gufo/pull/384)).
- Local: commit introducing this entry.
- Adaptation: independently implement `--cache-ram-bytes` through the existing
  pool admission callback. Zero preserves the fork's automatic budget; positive
  values lower it, clamped to the model's post-session allocation claim. Preserve
  Windows memory accounting, six entries per session, immutable host-prefix
  sharing and the snapshot/disk format. Omit upstream's 128-entry redesign,
  automatic 32 GiB ceiling and ability to raise the automatic budget: these are
  unnecessary for reducing retained memory. Add a per-preset GUI GiB control,
  defaulting older presets in memory without writing them on load.
- Areas: shared runner pool and inference loader, serve CLI, GUI settings/form,
  translations and fixtures; cache/server/GUI documentation. Other memory PRs
  are tracked separately in [todo.md](todo.md).
- Validation: Windows release and focused CPU builds pass; six cache/runner/
  scheduler/API suites, 35 GUI Python tests, 17 jsdom tests, formatting and docs
  pass. All 117 matched full-logit rows are exact with unchanged perplexity.
  Flash-Next IQ4_XS + Q8_0 MTP HTTP replay at 4096 context produces identical
  completions under refusing (1 byte) and fitting (1 GiB) caps, with RAM restore
  only under the fitting cap. See [validation identities and commands](docs/ram-snapshot-cap-validation.md).
  Linux CI, other quantizations and measured long-context Windows committed-memory
  savings remain unrun.
- Status: retained.

### 2026-10-08 — Short-request prefill fairness

- Source: scheduling portion of
  [gufo-org/gufo@def2ed1e904f912e69550344aa7c13a99f535d0e](https://github.com/gufo-org/gufo/commit/def2ed1e904f912e69550344aa7c13a99f535d0e)
  from [PR #470](https://github.com/gufo-org/gufo/pull/470), final head
  `5c6ebb4660b6712b18c6a2c20b3ee3d3e1c3cf2a` (squash
  `47b639159315fcdba17e6a144d67e273de9ead6e`).
- Local: commit introducing this entry; attention prerequisite
  `29a26847a4becc761bd692d6fd46a0f9eb4d1c8a`.
- Adaptation: give arrivals a turn before the last-served prefill repeats,
  preserve older waiting peers, bound work ahead of short prompts by the
  existing decode budget, and delay a ready first token for initial batch
  assembly only when the pending peer fits that budget. Keep wide alternating
  long-prefill turns and hand off after the current forward. Retain the
  2048-token engine default; omit prefetch, reader tuning, engine scratch reuse
  and upstream's 4096-token default.
- Areas: shared text-generation scheduler, four AR/MTP deterministic fixtures,
  explicit standard-library HTTP check and server documentation.
- Validation: pre-fix arrival regression reproduced; fresh Windows builds pass
  scheduler, device-loss, runner, OpenAI and HTTP tests (5/5). Actual C2 HTTP
  short output precedes long-prefill completion, with long progress and ongoing
  decode. All 23 serving sampling strategies retain exact replay/budgets, and
  the final release matches all 549 teacher-forced rows with unchanged
  perplexity. The 57-kernel production guard and assertion-build
  `kernel_resources_test` remain clean. See
  [qualification and identities](docs/prefill-pr470-qualification.md) for actual
  measurements and retained attention limitations. Linux build/CI, other
  models/quantizations and sustained arrival-load measurements are unrun.
- Status: scheduling retained. A later independent
  [4096-token chunk experiment](docs/plans/prefill-chunk-4096.md) makes 4096
  the fork's engine default; no additional #470 prefetch code is imported.

### 2026-10-08 — Flash-Next attention at the sparse budget

- Source: [gufo-org/gufo@f5401508e0352afbfc9d769414b15486ea100cdf](https://github.com/gufo-org/gufo/commit/f5401508e0352afbfc9d769414b15486ea100cdf)
  from [PR #470](https://github.com/gufo-org/gufo/pull/470), reviewed at final head
  `5c6ebb4660b6712b18c6a2c20b3ee3d3e1c3cf2a` (squash
  `47b639159315fcdba17e6a144d67e273de9ead6e`).
- Local: `29a26847a4becc761bd692d6fd46a0f9eb4d1c8a`.
- Adaptation: keep pre-budget queries on dense tiled attention when a retained
  prefix makes the existing 2048-token chunk cross the sparse budget. Run the
  sparse tail first, preserve refusal/fallback and last-only catch-up, and
  project the combined context once. Preserve projections, ranking and other
  arithmetic. Bump Flash-Next snapshots to version 16 to reject older state.
  Omit upstream checkpoint machinery absent from this fork and the broader
  prefill tuning deferred by this adaptation's scope.
- Areas: Flash-Next executor, snapshot compatibility, existing session chunk
  fixtures and cache documentation.
- Validation: reproduced 4096/1024 failure before the fix; fresh pinned Windows
  builds pass all 80 AR/MTP chunk comparisons, 270 unchanged-schedule full rows,
  549 teacher-forced hashes and unchanged perplexity. Analytic operators,
  snapshots, all 105 rollback prefixes, image-prefix continuation, batching and
  23 serving sampling strategies pass. The 57-kernel resource guard stays clean.
  [Qualification and identities](docs/prefill-pr470-qualification.md) also record
  remaining nonaligned indexer differences and excluded instrumented attempts.
  Linux build/CI and additional quantizations are unrun.
- Status: retained.

### 2026-10-07 — Clang 23 prefill scratch and spills

- Sources: [gufo-org/gufo@f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253](https://github.com/gufo-org/gufo/commit/f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253)
  (merged [PR #459](https://github.com/gufo-org/gufo/pull/459)); constituents
  [2f223d79a146a4ebf26a168f39b89dde8a96c050](https://github.com/gufo-org/gufo/commit/2f223d79a146a4ebf26a168f39b89dde8a96c050),
  [53912b8ffe45c9ce16fc04555718c85409bc32e1](https://github.com/gufo-org/gufo/commit/53912b8ffe45c9ce16fc04555718c85409bc32e1)
  and [9bccebc5e1d42a702820700736b2601cb1533e90](https://github.com/gufo-org/gufo/commit/9bccebc5e1d42a702820700736b2601cb1533e90).
- Local: `6244865bb4bc9cbcc703fa540d6354e56809af04` (kernel adaptation);
  mixed-Q4/Q5 capture and qualification in
  `be82fcdfdd82b7a8a69d5fbf3d4fe4c90d8ff632`; full Qwen Q8 qualification
  in the commit updating this entry.
- Adaptation: constrain blocked Qwen K-loop unrolling and W8A8 scheduling,
  disable SLP only in the two affected Qwen prefill compilation units, and
  select paired Flash-Next Q4/Q5 cache values from fixed register slots.
  Preserve this fork's existing Flash-Next W8A8 scheduling fence. Adapt the
  offload/metadata reader to Windows PE and Linux ELF, using the compiler's
  bundler and a focused zero-scratch/spill contract for 57 required gfx1151
  variants. Omit upstream's broad Linux baseline/allowance tables and comparison
  modes: this change guards the regression's affected families directly.
- Areas: Qwen blocked prefill kernels/CMake, Flash-Next routed GEMM,
  `tools/ci/check-kernel-resources.py`, CTest registration, boundary/operator
  fixtures and Qwen's full-logit capture diagnostic.
- Validation: fresh TheRock 10.0.0/clang 23 Windows builds; baseline has 17
  resource failures, candidate has none. Ten parser fixtures, both Qwen quant
  operator tests, routed WMMA operators, full Qwen target checks and Flash-Next
  prefill/session checks pass. All 189 Qwen full-logit rows and 549 Flash-Next
  raw-logit hashes match, with unchanged teacher-forced perplexity. Matched
  native performance results, identities, commands and diagnostic limits are
  in the [qualification record](docs/plans/prefill-spills-pr459.md). Additional
  UD-Q4_K_XL/Q8-MTP validation covers 47 Q4_K layer pairs and one Q5_K pair:
  loading, replay and all 270 full-logit rows pass with identical perplexity.
  Full Qwen27B Q8_0 also retains 189 byte-identical logit rows and perplexity,
  with the native target state/replay/rollback suite passing. Linux build/CI
  and clang 22 runtime validation are unrun.
- Status: retained.

### 2026-10-07 — Automatic disk staging without a fixed cap

- Source: [gufo-org/gufo@b39c530e70e87f4340e2230a155fd16d066d19f3](https://github.com/gufo-org/gufo/commit/b39c530e70e87f4340e2230a155fd16d066d19f3)
  (PR #473, merged; constituents
  [4ab4b9152189e1956a7c9959c9e4ae159b2bd215](https://github.com/gufo-org/gufo/commit/4ab4b9152189e1956a7c9959c9e4ae159b2bd215)
  and [7f6ea6e1d686d863859eb3a248ce9088fd2cd07e](https://github.com/gufo-org/gufo/commit/7f6ea6e1d686d863859eb3a248ce9088fd2cd07e)).
- Local: commit introducing this entry.
- Adaptation: remove the fixed 1 GiB automatic staging ceiling in this fork's
  existing bounded disk store. Zero selects `min(disk capacity,
  HostSnapshotBudgetBytes()/4)`; explicit limits and pending-capture accounting
  retain their behavior. Update local CLI, server/cache docs and GUI hints.
  No broader cache port is needed. Capture reuse, the `min_step` policy and early
  disk admission remain separate future improvements. RAM checkpoints, images,
  reasoning state, persistence format and cache fallback are unchanged.
- Areas: `src/cli/serve/{continuation_disk_store.cpp,text_model_runner.hpp,serve.cpp}`,
  disk-store/CLI tests, `docs/{SERVER,KV-CACHE}.md`, GUI template and README.
- Validation: fresh Windows production and CPU builds; disk-store, runner and
  CLI CTest pass, as do release CLI checks, 34 GUI tests, clang-format 21.1.8
  and documentation checks. Allocation-free admission above 1 GiB passes here
  and fails with the old calculation; low-memory hosts skip that branch.
  Qwen27B auto stores a 1,111,625,411-byte checkpoint, drains on shutdown,
  restores after restart and matches cold/live continuation answers. Old auto
  and explicit 1 GiB reject it with correct fallback. Numerical capture has
  27/27 identical full-logit rows and unchanged perplexity; this uses unchanged
  model code, independently of the disk HTTP checks. See
  [qualification and limits](docs/plans/disk-staging-pr473.md). Linux CI is
  unrun; the Windows symlink fixture lacks privileges.
- Status: retained.

### 2026-10-05 — Bounded inferred Qwen argument types

- Sources: [gufo-org/gufo@29fb0b4eec70f49088bb5126074e6f10be17a0d1](https://github.com/gufo-org/gufo/commit/29fb0b4eec70f49088bb5126074e6f10be17a0d1)
  and [gufo-org/gufo@b45567a918b281bdce63e80dec3e6184ccdb9c25](https://github.com/gufo-org/gufo/commit/b45567a918b281bdce63e80dec3e6184ccdb9c25)
  (PR #441 final shape).
- Local: d7136e78b7d4417900bf1dab34df63bcc5ff2170.
- Adaptation: finite values and implicit object/array/string shapes supply
  value-kind hints to the existing bounded resolver. Enum members consume its
  128-node budget. Explicit types and finite values precede implicit shapes;
  applicable hints intersect. Preserves string preference for mixed unions,
  pattern/additional-property hints, local reference bounds and unsupported
  fallback. No schema validator, generation grammar or inference changes.
- Areas: `src/cli/serve/openai_chat.cpp`, `tests/cli/openai_chat_test.cpp`,
  `docs/SERVER.md`.
- Validation: numeric enum fixture fails before adaptation. Fresh Windows CPU
  `openai_chat_test` and `http_server_test` pass, covering finite JSON kinds,
  inferred containers, JSON-owned protocol tags, numeric-looking strings,
  mixed/conflicting hints, applicators and enum budget exhaustion in both APIs
  and response modes. Changed C++ files pass clang-format 21.1.8 and docs checks
  pass; the delimiter entry records inherited whole-tree format failures.
  Final [qualification](docs/plans/tool-parser-pr441.md) records the production
  Windows build, exact equality for 66 full-logit rows, unchanged perplexity,
  and AR/MTP enum/object calls, continuations and cache retries. Linux CI is
  unrun; real-model generation did not exercise the two other parser edges.
- Status: retained.

### 2026-10-05 — Exact declared Qwen parameter names

- Sources: [gufo-org/gufo@29fb0b4eec70f49088bb5126074e6f10be17a0d1](https://github.com/gufo-org/gufo/commit/29fb0b4eec70f49088bb5126074e6f10be17a0d1)
  and [gufo-org/gufo@b45567a918b281bdce63e80dec3e6184ccdb9c25](https://github.com/gufo-org/gufo/commit/b45567a918b281bdce63e80dec3e6184ccdb9c25)
  (PR #441 final shape).
- Local: e5dd53adea21de8af18766778d9f262b58b79985.
- Adaptation: scanner and decoder use one spelling decision. Named root
  properties, including bounded local root references, retain surrounding
  whitespace and use the matching type hint. Unknown spellings retain legacy
  trimming. Existing pattern/additional-property hints, duplicate rules and
  thinking boundaries are unchanged; no grammar infrastructure is imported.
- Areas: `src/cli/serve/openai_chat.cpp`, `tests/cli/openai_chat_test.cpp`,
  `docs/SERVER.md`.
- Validation: the new spaced-name fixture fails before adaptation. Fresh Windows
  CPU `openai_chat_test` and `http_server_test` pass, including exact/legacy
  spellings, scalar/JSON type hints, root references and cycles in both APIs
  and response modes, with byte-wise chunks. Changed files pass clang-format
  21.1.8 and documentation checks pass; the delimiter entry records inherited
  whole-tree format failures. Final
  [qualification](docs/plans/tool-parser-pr441.md) records the production build
  and exact GPU numerical equality. The model emitted a trimmed name, leaving
  this edge covered by deterministic fixtures. Linux CI is unrun.
- Status: retained.

### 2026-10-05 — Canonical Qwen parameter delimiters

- Source: [gufo-org/gufo@b45567a918b281bdce63e80dec3e6184ccdb9c25](https://github.com/gufo-org/gufo/commit/b45567a918b281bdce63e80dec3e6184ccdb9c25)
  (PR #441; final merge
  [21d6e64f137f6bcbc5a8bf63f900cab648188df7](https://github.com/gufo-org/gufo/commit/21d6e64f137f6bcbc5a8bf63f900cab648188df7)).
- Local: ba1349e502ee45a83d8f42de4c1e2bcd2b9bb7b8.
- Adaptation: the consolidated incremental scanner and argument decoder share
  complete canonical delimiter recognition, including CRLF and split suffixes.
  Inline tags and closing-tag-like text remain data. Compact legacy delimiters,
  JSON string ownership, interrupted-call recovery and the explicit thinking
  boundary are retained. Omitted upstream grammar, implicit reasoning recovery,
  mask-cache and tokenization changes; this fork has no grammar fallback and
  already preserves literal-token cache prefixes.
- Areas: `src/cli/serve/openai_chat.cpp`, `tests/cli/openai_chat_test.cpp`,
  `docs/SERVER.md`.
- Validation: the new literal-delimiter fixture fails on the pinned fork
  `0a2b9fd65d5927888b6146fb07dbf9619984c505`. Fresh Windows CPU
  `openai_chat_test` and `http_server_test` pass after adaptation, covering both
  APIs, buffered/streamed output, every two-piece split and byte-wise chunks,
  LF/CRLF and empty strings. Changed C++ files pass clang-format 21.1.8;
  the initial full formatting check found two inherited violations in
  `http_server.cpp` and `openai_chat.hpp`, reproduced from the pinned base.
  A separate whitespace cleanup resolved both; all 485 C++ files now pass.
  Documentation checks pass. Final
  [qualification](docs/plans/tool-parser-pr441.md) records the production build,
  exact GPU numerical equality and AR/MTP tool/replay checks. The model omitted
  the literal closing-tag-like line, leaving this edge covered by deterministic
  fixtures. Linux CI is unrun.
- Status: retained.

### 2026-10-05 — Messages reasoning fields (Claude Code compat)

- Sources: [gufo-org/gufo@56383be0718ea0ce203fc42a55581304f0bbc253](https://github.com/gufo-org/gufo/commit/56383be0718ea0ce203fc42a55581304f0bbc253)
  (#405) and [gufo-org/gufo@c3f4a9c26031b04f34b84a96f4d8cfc8f4337ad2](https://github.com/gufo-org/gufo/commit/c3f4a9c26031b04f34b84a96f4d8cfc8f4337ad2)
  (#428, parser refactor), landed as one change.
- Local: 0ce114890049c90a728281dcfd3a2a648f9fa150 (manual port; both
  cherry-picks conflicted throughout).
- Adaptation: `/v1/messages` accepts `thinking.type` (`enabled`, `adaptive`,
  `disabled`) with `budget_tokens`/`display` validation, plus
  `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`, never enabling
  thinking, other members rejected) in the #428 end-state shape
  (`ReadMessagesOutputConfig` returning endpoint errors;
  `ParseReasoningEffortName` moved to server scope and declared in the hpp).
  `adaptive` keeps the server default via the fork's optional reasoning
  fields. Omitted thinking-block response rendering (this endpoint returns
  text only, as before) and the `cache_growth.py` harness tweak (functional
  suite removed in this fork); effort-consistency note kept for prefix reuse.
- Areas: `src/cli/serve/http_server.cpp`, `src/cli/serve/openai_chat.cpp`,
  `src/cli/serve/openai_chat.hpp`, `docs/SERVER.md`,
  `tests/cli/http_server_test.cpp`.
- Validation: Windows `cpu-test` builds clean; `http_server_test` (incl. new
  adaptive/effort acceptance and nine invalid-field cases) and
  `openai_chat_test` pass; `check-docs.py` passes (72 files, 420 links).
  `check-format.py` unrun — no repo-compatible `clang-format` on PATH. Linux
  `pr` suite and real-model smoke unrun.
- Status: retained.

### 2026-10-05 — Streaming failures carry the real error message

- Source: [gufo-org/gufo@bf605399477b694c19adb6313952ecbf0986d8be](https://github.com/gufo-org/gufo/commit/bf605399477b694c19adb6313952ecbf0986d8be)
  (#385).
- Local: 614ed35a (manual port; cherry-pick aborted after three-way
  conflicts).
- Adaptation: chat `StreamingResponse` and Responses `CreateOpenAiResponse`
  emit `error.what()` with a `"generation failed"` fallback for empty
  messages, keeping the fork's `write` naming and the Responses
  `server_error` code (the Responses wire enum stays closed; the runtime code
  remains log-only). Omitted the `/v1/completions` streaming hunk: this fork
  rejects streamed raw completions before admission (400) and removed that
  streaming path, so there is no counterpart site. Test fake gains the
  upstream empty-message failure mode; Responses and chat streaming failures
  now assert wire messages for both modes.
- Areas: `src/cli/serve/openai_chat.cpp`,
  `tests/cli/http_server_test.cpp`.
- Validation: Windows `cpu-test` builds clean; `http_server_test` and
  `openai_chat_test` pass. `check-format.py` unrun — no repo-compatible
  `clang-format` on PATH. Linux `pr` suite unrun.
- Status: retained.

### 2026-10-05 — Sampling block-skip for unselectable vocabulary ranges

- Source: [gufo-org/gufo@b945d0afdb26e4b790b8cb260762105e167ee1a3](https://github.com/gufo-org/gufo/commit/b945d0afdb26e4b790b8cb260762105e167ee1a3)
  (#415).
- Local: e8074c37 (`git cherry-pick -x` with manual conflict resolution).
- Adaptation: `BlockMaximum`/`kSkipBlock` and the `PrepareSelected` refactor
  (block-skipped top-k heap, shared penalty walk, full-sort/top-p
  consolidation) imported verbatim. `SampleGreedy` keeps the fork's
  `AdjustedLogit` per-index path instead of upstream's inline penalty walk;
  the block-skip gate still uses the sorted penalty iterator, which is exact
  because unpenalized blocks satisfy `adjusted == (double)logit` and ties
  never replace the earlier token. Test includes keep the fork's `<numeric>`
  and `<random>` alongside upstream's `<span>`. Fork's constraint removal and
  `DistributionFromTop`/`FinishSelected` split are untouched.
- Areas: `src/core/sampling.cpp`, `tests/cli/logit_sampler_test.cpp`.
- Validation: Windows `cpu-test` builds clean; `logit_sampler_test` passes
  including the new `TestBlockSkippedTopKAgainstFullSort` (70k-row reference
  comparison across five configs with penalties and non-finite entries);
  `text_generation_scheduler_test` (relinked against the changed core lib)
  and `qwen38_flash_next.mtp_sampling` pass. `check-format.py` unrun — no
  repo-compatible `clang-format` on PATH. Linux `pr` suite and real-model
  full-logit/perplexity comparison unrun; the change is selection-identical
  by construction plus CPU equivalence coverage.
- Status: retained.

### 2026-10-05 — Reclaim output capacity from abandoned streams

- Source: [gufo-org/gufo@d85fb0fcd3fb906ea845f6e1404222d686b74e07](https://github.com/gufo-org/gufo/commit/d85fb0fcd3fb906ea845f6e1404222d686b74e07)
  (#394).
- Local: 49852db5a6b020a25ff2fadc71b7a79b84ed80a0 (`git cherry-pick -x`, no
  manual edits).
- Adaptation: verbatim import. `ScheduledRequest` releases its remaining
  `buffered_output_bytes` charge when the last owner drops a completed but
  never-consumed stream, so a replacement request can reuse the capacity.
  Nothing omitted; no fork-specific adjustment needed — the fork's device-loss
  probe lives in `CompleteFailure`, far from the destructor site, and the
  consume path already releases per piece, so the destructor only returns the
  remainder.
- Areas: `src/cli/serve/text_generation_scheduler.cpp`,
  `tests/cli/text_generation_scheduler_test.cpp`.
- Validation: Windows `cpu-test` build of `text_generation_scheduler_test`
  passes, including the two new upstream cases
  (`TestCompletedAbandonedStreamReleasesOutputBudget`,
  `TestCompletedStreamOutlivesScheduler`); direct binary run reports all tests
  passed. `check-format.py` unrun — no repo-compatible `clang-format` on PATH
  and an unrelated version could rewrite files; the change is verbatim
  upstream, format-clean by construction. Linux `pr` suite unrun. No logit or
  perplexity check: capacity accounting only, no inference arithmetic change.
- Status: retained.

### 2026-10-03 — DeepSeek call-block separator replay

- Source: [gufo-org/gufo@43e8351daa37f3115249372809b5a4bd0bc3d481](https://github.com/gufo-org/gufo/commit/43e8351daa37f3115249372809b5a4bd0bc3d481)
  (#404).
- Local: commit introducing this entry.
- Adaptation: the consolidated parser retains at most two trailing newlines
  until the next opener is known. The canonical DeepSeek call separator belongs
  to framing once a call is attempted; literal mentions and empty examples keep
  their original bytes. Qwen, prose, code, suffixes and other whitespace retain
  their contracts. No separate buffered/streamed separator implementations.
- Areas: shared API output parser and framing fixtures; completed real-model
  replay harness and validation record.
- Validation: fresh five-target Windows CPU suite and production GPU build pass.
  Both APIs and response modes agree at every two-piece and byte-by-byte split.
  Qwen27B candidate passes 70 HTTP checks, including AR/DFlash2, literal history,
  user images and Responses image tool results. Forty-two matched HTTP
  signatures and all 27 full-logit rows match their recorded baselines exactly;
  perplexity is unchanged. Formatting/docs checks pass. Real DeepSeek and Linux
  runtime validation remain unrun; see the
  [validation record](docs/plans/history-replay.md).
- Status: retained.

### 2026-10-03 — Literal prompt control-token handling

- Source: history replay problem investigated alongside
  [gufo-org/gufo@0564df5ffcabb24b6b70df3a17f7fd8cce1718bb](https://github.com/gufo-org/gufo/commit/0564df5ffcabb24b6b70df3a17f7fd8cce1718bb)
  (#400 reviewed head); implementation is local.
- Local: commit introducing this entry.
- Adaptation: renderer-owned literal byte ranges suppress special-token matches
  in message/reasoning data, tool schemas, historical names and argument values.
  Tokenization retains one BPE pass between structural controls, preserving
  ordinary merges. Both direct rendering and actual HTTP `vision::Prepare`
  honor these ranges, including plain requests and real image expansion.
  Bump Qwen formatter identities. Omitted output sanitization and speculative
  EOS-policy changes; data is retained verbatim.
- Areas: Qwen renderer/tokenizer, HTTP vision preparation, compatibility identity,
  deterministic prompt/image fixtures and explicit real-model replay checks.
- Validation: fresh five-target Windows CPU suite and production GPU build pass.
  Qwen27B AR/DFlash2 typed and literal replay checks pass; first canonical warm
  replay reuses 413 tokens versus baseline 353. All 27 full-logit rows and
  perplexity match exactly. Vision and matched HTTP checks are documented in the
  [validation record](docs/plans/history-replay.md). Linux execution is unrun.
- Status: retained.

### 2026-10-03 — Typed argument history replay

- Source: [gufo-org/gufo@f51d33bd553b42f57cc2a291f2584933c956614f](https://github.com/gufo-org/gufo/commit/f51d33bd553b42f57cc2a291f2584933c956614f)
  (#404).
- Local: commit introducing this entry.
- Adaptation: serialize non-string historical arguments at the Qwen/DeepSeek
  rendering boundary with ordered UTF-8 JSON and reference separator spacing.
  Strings and compact API JSON retain their existing representation. This also
  covers internal template callers. Bump persistent formatter identities;
  payload formats and inference arithmetic are unchanged. Omitted upstream's
  union-grammar work because this fork has no corresponding grammar fallback.
- Areas: shared JSON serializer, both model formatters, compatibility identities,
  CPU fixtures and the explicit `history_replay_test.py` real-model harness.
- Validation: fresh Windows JSON, Qwen/DeepSeek template, API and HTTP tests pass
  (five targets). Qwen27B GPU comparison and HTTP checks pass, with exact
  full-logit equality and improved canonical call replay. Results are recorded in
  [history replay validation](docs/plans/history-replay.md).
- Status: retained.

### 2026-10-03 — Consolidated tool output framing

- Source: [gufo-org/gufo@c33e050eced6389852617994fe7349367df4c900](https://github.com/gufo-org/gufo/commit/c33e050eced6389852617994fe7349367df4c900)
  (#393), [2c6a1064f39a4d3ea0b8d92efea0beedf18150f1](https://github.com/gufo-org/gufo/commit/2c6a1064f39a4d3ea0b8d92efea0beedf18150f1)
  (#396), [d91674a4dd8479a6e6c7044e0b784099ff25b31f](https://github.com/gufo-org/gufo/commit/d91674a4dd8479a6e6c7044e0b784099ff25b31f)
  (#397 draft), and [6718dba3293a0e89696029ccf036bfb4646dd251](https://github.com/gufo-org/gufo/commit/6718dba3293a0e89696029ccf036bfb4646dd251)
  (#391 draft).
- Local: commit introducing this entry.
- Adaptation: one incremental framing parser for both APIs and response modes,
  request-owned native format, JSON quote/escape ownership, bounded DSML
  envelopes, preserved suffix prose/code examples and explicit content-mode
  thinking tags. Retains the fork's argument/type decoding and error contracts.
  Omitted upstream grammar infrastructure and automatic format switching;
  history rendering, DeepSeek separators and speculative/history accounting
  remain separate. No broad delimiter sanitization was imported.
- Areas: API output parser, generation request/runner metadata, regression
  fixtures and server contracts.
- Validation: fresh Windows CPU API/HTTP tests and production GPU build pass;
  all 174 full-logit rows and three binary dumps match the baseline exactly,
  with identical perplexity. Eight matched HTTP cases and four required-tool
  cases pass. Formatting and docs checks pass. Linux execution and real
  DeepSeek/Qwen27B model checks remain unrun; see the
  [validation record](docs/plans/tool-output-parser.md).
- Status: retained.

### 2026-10-02 — Exit immediately after confirmed text-device loss

- Source: [gufo-org/gufo@7d893a020372502757933ec80231927fc9d25a96](https://github.com/gufo-org/gufo/commit/7d893a020372502757933ec80231927fc9d25a96)
  (PR #390).
- Local: commit introducing this entry.
- Adaptation: retained the preallocated four-byte HIP probe and five-second
  polling deadline for Qwen, Flash-Next and DeepSeek. Runs detection before
  request invalidation; a serve-owned callback writes the original failure
  reason and exits 75 without model teardown. Pending probes and classified
  request errors keep their existing behavior. Omitted sticky loss state,
  HTTP/health status changes, new wire errors, signal shutdown and watchdog:
  immediate process exit closes clients and avoids dead-device cleanup.
- Areas: text runners/scheduler, serve process policy, CPU/process fixtures,
  Windows and Linux hosted CPU selections, and server documentation.
- Validation: fresh Windows GPU build and four focused CPU tests pass; the
  process control exits 76 at invalidation and the production handler exits 75
  first. GUI tests pass (34). All 174 full-logit rows and perplexity match the
  recorded baseline exactly; eight real-model HTTP cases match, including
  streaming, history reuse and concurrency. Formatting and docs checks pass;
  see the [validation record](docs/plans/device-loss-pr390.md). Actual
  driver-reset and Linux execution remain unrun.
- Status: retained.

### 2026-10-01 — Multiline edit boundaries and malformed-call diagnostics

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373 multiline edit regressions).
- Local: 55ff2aff89dfa6c01a45a1b15c13f55fa09b9b8c.
- Adaptation: delimiters inside JSON strings are data in native Qwen/DeepSeek
  array/object arguments; thinking markers after an enabled tool opener do not
  start inline reasoning. DeepSeek raw string quotes retain native semantics.
  Added a fork-specific, non-retryable `malformed_tool_call` 502/SSE contract for
  complete framed attempts rejected at EOS. Stops, limits and cancellation omit
  incomplete calls normally, including required choice. Retains the fork's
  thinking boundary, duplicate rules and interrupted-call recovery. No blanket
  control repair, schema enforcement, automatic JSON routing or loop suppression.
- Areas: parser, backend error contract, API/HTTP fixtures and server contracts.
- Validation: multiline fixture failed before the boundary adaptation. Fresh
  Windows CPU `openai_chat_test`, `http_server_test`, `qwen_chat_template_test`
  and `ds4.template` passed. Covers both APIs buffered/streaming, mixed valid/bad
  output, escaped/raw newlines, literal Qwen/DSML/thinking delimiters, native
  string quotes, non-retryable buffered errors, SSE failure after headers and
  required calls at token limits. The [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the partial real-model results and remaining failures. Linux CI is
  outstanding.
- Status: retained.

### 2026-10-01 — Bounded Qwen declared-type recovery

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373 typed wildcard regression and native-schema review).
- Local: 5935c53e99b451a1cd26a37c7cf47aa5391d036b.
- Adaptation: parser-only type hints from named properties, all matching
  patterns, additional-property schemas and bounded local references. Uses
  existing ICU with `ICU::i18n` linkage and a documented portable regex subset.
  Ambiguous/unsupported hints retain text. This is an independent adaptation,
  not upstream's wildcard JSON fallback or a schema validator. Does not import
  automatic protocol switching, constraints or inference changes.
- Areas: `CMakeLists.txt`, API parser/fixtures and server contracts.
- Validation: baseline fixture reproduced integer `x_count` becoming a string.
  Fresh Windows CPU `openai_chat_test` and `http_server_test` pass; coverage
  includes all JSON types, intersections, precedence, Unicode/escaped keys,
  reference chains/cycles, bounded lookup and guidance fallback. Pinned ICU 78.3
  Windows import libraries and `icuin78.dll` linked/staged successfully.
  Real-model wildcard probes pass in all four tested modes; the
  [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the partial qualification and remaining failures. Linux CI is
  outstanding.
- Status: retained.

### 2026-10-01 — Historical function-name preservation

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373).
- Local: 5a95703c90afac4a448047e28044ed64a4d3bb2c.
- Adaptation: shared historical-function parsing for Chat Completions and
  Responses. Preserves non-empty string names except embedded NUL and requires
  JSON-object arguments. Declaration rules and generated-call allowlists remain
  intact. The native renderers retain historical names verbatim, including
  delimiter-bearing names; this is history preservation, not escaping or new
  invocation authorization. No upstream grammar or renderer changes imported.
- Areas: API parser, API fixtures, Qwen/DeepSeek template fixtures and server
  contracts.
- Validation: new history fixture failed against the previous parser. Fresh
  Windows CPU API and Qwen/DeepSeek template targets passed after adaptation;
  existing stateless grouping, call/result identity and image-order fixtures
  remain covered. Real-model Unicode history probes pass; the
  [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the failed exact-file agent goals. Linux CI is outstanding.
- Status: retained.

### 2026-10-01 — Disabled-tool marker delivery

- Source: [gufo-org/gufo@594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e)
  (PR #373).
- Local: f997a39f14da75719a407ee99931736cdf1b4a4b.
- Adaptation: one recognition decision for buffered and streaming output in
  both APIs. Disabled tool markers remain text/reasoning and their prefixes
  stream immediately. Retains the fork's required thinking boundary and UTF-8
  decoding. Does not import upstream constraints, prompting or sampling changes.
- Areas: `src/cli/serve/openai_chat.cpp`, `tests/cli/openai_chat_test.cpp`,
  `docs/SERVER.md`.
- Validation: baseline Windows CPU `openai_chat_test` and `http_server_test`
  passed at `c4dcca536af5dc73998f24b28e548985e7f91efc`; new deterministic
  disabled-marker fixture reproduced delayed prefixes, then passed after the
  adaptation. Both APIs, buffered/streaming, disabled declarations, reasoning,
  byte-split markers/UTF-8 and immediate callback delivery are covered.
  Real-model literal probes pass; the
  [implementation record](docs/plans/tool-calling-pr373.md#implementation-record)
  retains the partial qualification. Linux CI is outstanding.
- Status: retained.

### 2026-10-01 — Conversation history-edit checkpoints

- Source: [gufo-org/gufo@0c350ed3ef5f8db9783385071ff05eec6bd687f1](https://github.com/gufo-org/gufo/commit/0c350ed3ef5f8db9783385071ff05eec6bd687f1)
  and [gufo-org/gufo@840d3736012ebeb123472b2dc8ca39411084b05e](https://github.com/gufo-org/gufo/commit/840d3736012ebeb123472b2dc8ca39411084b05e)
  (PR #362 branch work), plus grid deduplication from
  [gufo-org/gufo@d8475862a1872ddbadefa281e0aee54350f18e9f](https://github.com/gufo-org/gufo/commit/d8475862a1872ddbadefa281e0aee54350f18e9f)
  (PR #358).
- Local: 733aee8a44ba0ccaaae9b4448e8b316f6265611b.
- Adaptation: four bounded intermediate RAM checkpoints on a 2048-token grid
  and six retained entry slots per execution session, within the existing byte
  budget. Flash-Next snapshot pages are pre-faulted with bounded fault workers
  before huge-page advice via the existing Windows madvise layer. Preserves the
  fork's prefix-scoped image identity and the #358 warm-turn stable
  boundary/source protection. The upstream history-edit and rewind regression
  was adapted to a standalone stdlib HTTP check sharing the existing
  cache-growth transport; the unrelated upstream functional SDK framework was
  not imported.
- Areas: `src/cli/serve/text_model_runner.cpp`,
  `src/models/qwen38_flash_next/engine.cpp`,
  `tests/cli/text_model_runner_test.cpp`, `tests/tools/cache_edits_test.py`,
  `tests/tools/cache_growth_test.py`, `docs/KV-CACHE.md`, `docs/SERVER.md`,
  `docs/TESTING.md`, `docs/models/qwen3.8-27b/EXPERIMENTS.md`,
  `docs/models/qwen3.8-flash-next/EXPERIMENTS.md`.
- Validation: CPU assertions and the `cache_edits_test.py` HTTP check were
  added in-commit; run results were not verified while writing this entry.
- Status: retained.

### 2026-10-01 — Warm-turn stable checkpoints

- Source: [gufo-org/gufo@d8475862a1872ddbadefa281e0aee54350f18e9f](https://github.com/gufo-org/gufo/commit/d8475862a1872ddbadefa281e0aee54350f18e9f)
  including [gufo-org/gufo@492e5e1a948c5ac5456f850a7a128d0bee01e94c](https://github.com/gufo-org/gufo/commit/492e5e1a948c5ac5456f850a7a128d0bee01e94c)
  (PR #358 branch work), plus only the `PublishSnapshot` source-preservation
  API prerequisite from PR #362
  ([gufo-org/gufo@0c350ed3ef5f8db9783385071ff05eec6bd687f1](https://github.com/gufo-org/gufo/commit/0c350ed3ef5f8db9783385071ff05eec6bd687f1)).
- Local: 031d2891acf8ecc4a3027ffcccd84407c411b14b.
- Adaptation: retains the turn's own stable boundary after freezing its reused
  frontier, prefers replacement of incompatible branch tails, and guards
  against concurrently replaced sources. Uses three checkpoint entries per
  session (frontier, stable boundary, complete prompt) with unchanged byte
  budgets and execution-state count; preserves the fork's prefix image identity
  and Windows persistence behavior. Deferred the intermediate history
  checkpoints (later covered by `733aee8`), the six-entry layout and
  Flash-Next allocation changes. CPU regressions were ported and cache-growth
  functional coverage was adapted into an explicit stdlib-only real-model
  check instead of the absent upstream SDK framework.
- Areas: `src/cli/serve/continuation_cache.cpp`,
  `src/cli/serve/continuation_cache.hpp`,
  `src/cli/serve/text_model_runner.cpp`,
  `tests/cli/continuation_cache_test.cpp`,
  `tests/cli/text_model_runner_test.cpp`, `tests/tools/cache_growth_test.py`,
  `docs/KV-CACHE.md`, `docs/SERVER.md`, `docs/TESTING.md`.
- Validation: ported CPU regressions and the `cache_growth_test.py` check were
  added in-commit; run results were not verified while writing this entry.
- Status: retained; extended by `733aee8` (intermediate checkpoints).

### 2026-09-29 — Stable reasoning and image boundaries

- Source: [gufo-org/gufo@8783ccbb2c4ed6ff5fc2f6e19d774ffd04eeea6e](https://github.com/gufo-org/gufo/commit/8783ccbb2c4ed6ff5fc2f6e19d774ffd04eeea6e)
  (PR #301).
- Local: dbdc36cb2fb14f16227babe0528822832c257944.
- Adaptation: partial integration of the stable-boundary changes, preserving
  this fork's image-prefix reuse and Windows tuning. Omitted upstream disk
  image-prefix indexing and Qwen27B executor cancellation changes.
- Areas: `src/cli/serve/inference_backend.cpp`,
  `src/cli/serve/text_model_runner.cpp`, `src/models/qwen/chat_template.cpp`,
  `src/models/qwen/chat_template.hpp`, `src/models/qwen/vision/prompt.cpp`,
  `src/models/qwen/vision/prompt.hpp`, `tests/cli/text_model_runner_test.cpp`,
  `tests/models/qwen27b/vision_test.cpp`,
  `tools/serving/check-continuation.py`, `docs/SERVER.md`.
- Validation: assertions added in `tests/cli/text_model_runner_test.cpp` and
  `tests/models/qwen27b/vision_test.cpp` in-commit; run results were not
  verified while writing this entry.
- Status: retained.

### 2026-09-29 — Fallback and full-prompt checkpoints

- Source: [gufo-org/gufo@c362049c59e11a8fb998f5b2e0df2cfb8fac4c9c](https://github.com/gufo-org/gufo/commit/c362049c59e11a8fb998f5b2e0df2cfb8fac4c9c)
  (PR #281).
- Local: d25f30bf309cae50201495deba84c5991edbed22.
- Adaptation: retains the earlier fallback plus a complete prompt checkpoint;
  exact retries can avoid prefill without extra model sessions.
- Areas: `src/cli/serve/continuation_cache.cpp`,
  `src/cli/serve/continuation_cache.hpp`,
  `src/cli/serve/continuation_disk_store.cpp`,
  `src/cli/serve/continuation_disk_store.hpp`,
  `src/cli/serve/text_model_runner.cpp`,
  `tests/cli/text_model_runner_test.cpp`,
  `tests/models/qwen/tokenization/chat_template_test.cpp`,
  `tools/serving/check-continuation.py`, `docs/SERVER.md`.
- Validation: coverage added in `tests/cli/text_model_runner_test.cpp` and
  `tests/models/qwen/tokenization/chat_template_test.cpp` in-commit; run
  results were not verified while writing this entry.
- Status: retained.

### 2026-09-26 — Bounded disk staging with skipped-snapshot reporting

- Source: [gufo-org/gufo@d9a84f13f35d1f98da22886a12eb25dc7062e392](https://github.com/gufo-org/gufo/commit/d9a84f13f35d1f98da22886a12eb25dc7062e392)
  (PR #279, shared history with `official/main`).
- Local: same SHA (present verbatim, not an adaptation).
- Adaptation: none.
- Areas: `src/cli/serve/continuation_disk_store.cpp`,
  `src/cli/serve/continuation_disk_store.hpp`,
  `src/cli/serve/inference_backend.cpp`, `src/cli/serve/inference_backend.hpp`,
  `src/cli/serve/serve.cpp`, `src/cli/serve/text_model_runner.cpp`,
  `src/cli/serve/text_model_runner.hpp`,
  `tests/cli/continuation_disk_store_test.cpp`, `tests/cli/serve_test.py`,
  `docs/SERVER.md`.
- Validation: `tests/cli/continuation_disk_store_test.cpp` coverage added
  upstream in-commit; run results were not verified while writing this entry.
- Status: retained.

## Reviewed, not imported

Record upstream candidates examined against this fork and deliberately not
imported, newest first. These are decisions and their evidence, not imports;
revisit when the blocking condition changes.



## Model-specific provenance

- [DeepSeek V4 Flash](src/models/deepseek_v4_flash/UPSTREAM.md): imported engine
  revision, integration boundaries, and update policy.
- [Qwen-Image-2.1](src/models/qwen_image_21/UPSTREAM.md): pinned model and reference
  implementation contracts.
