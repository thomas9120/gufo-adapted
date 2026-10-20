# Qwen3.8 Flash-Next quality

**All 63 measured tg128 requests match fresh AR completions:** 21 AR,
21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. Latest measurements: September 20–23, 2026.

| Check | Result |
| --- | --- |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 23 AR/MTP configurations; shared FP64 target filtering/CDF, p/q acceptance and residual correction pass |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens; past 2048, a chunk boundary inside an absolute four-position group can differ ([causes](../../plans/iq-experts-prefill.md#natural-text-check)) |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the full Q8 head's
top 64 logits; upstream draft-sampler equivalence is not claimed.

## Experimental IQ F16 prefill

October 19, 2026, native Windows/gfx1151: candidate `ec31961208de` versus
`0817d23b6974`, same TheRock 10.0.0/clang 23 and artifacts. The existing
qualification gate reports **FAIL for UD-IQ4_XS** and **PASS for UD-Q4_K_XL**.
Both pass four CPU contracts, eleven GPU operator tests and all four model/state
suites. Q4 retains all 999 complete vocabulary rows and perplexities exactly.

IQ deliberately changes routed activation arithmetic from INT8 quantization
to F16. All 999 baseline logit rows change; normal-prefill fixture perplexity
increases **3.3522%**, from 1.2427840354 to 1.2844440694, and 314/324 top-1
predictions agree. AR/MTP halves are byte-identical within the candidate.
The increase comes from the 22 of 144 labels where the repeated pattern
wraps into a string-initial token the model never sees naturally (+26%);
the other 122 labels improve 0.27%. On 26,572 wikitext positions through
the prefill path, the build change is +0.23% at p8 and -0.07% at p5. That
is within the +0.22% the baseline alone moves between those widths, and
every interval includes zero.

Independent F64 operators pass their unchanged limits, including real expert
dimensions and tiny/ragged batches. On three short matched CPU-reference
fixtures, top-1 agreement improves from 51/52 to 52/52 and mean KL from
0.00439308 to 0.00250885; maximum KL increases from 0.03179 to 0.03374.
Printed reference KL values have five decimal places.

The candidate remains experimental. No golden or tolerance was changed, and
the model has no declared numerical acceptance limits for this transition.
Linux, unquantized-teacher qualification and image qualification remain
unrun. [Scope, commands and interpretation](../../plans/iq-experts-prefill.md)
and [compact identities/results](artifacts/iq-prefill-f16-candidate.json).

## Vision

September 29 Windows image-prefix cache qualification: a text prefix followed
by two 512x512 synthetic screenshots retains byte-identical full logits and
teacher-forced perplexity versus the recorded pre-change baseline, with AR and
MTP. The focused test checks live extension, snapshot restore, greedy/seeded
decoding and rejection of a changed earlier image. This is cache consistency
validation, not additional encoder quality qualification.
[Raw hashes and HTTP measurements](artifacts/image-prefix-cache.json).

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target qwen38_flash_next_image_prefix_test -Jobs 4
.\build\gpu-test\qwen38_flash_next_image_prefix_test.exe "$MODEL" "$MTP" "$MMPROJ" screenshot-a.png screenshot-b.png
```

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Reproduce

### Windows inference-qualification workflow exercise

October 18, 2026: clean builds at `24c0a99f8576` and `fe8469b56630` exercise
the qualification runner on the snapshot payload version 19 -> 20 fix, with
unchanged inference arithmetic. Native Windows/gfx1151, TheRock 10.0.0/clang
23, the same four-shard UD-Q4_K_XL target and Q8_0 MTP sidecar:

- The runner reports **PASS**, with distinct binaries, matched artifacts and
  no control or mismatch overrides.
- All **999 complete vocabulary rows** match exactly: 675 teacher-forced rows
  across schedules `1:2,4,8:p8` and 324 normal-prefill rows in AR/MTP through
  32K. Every row contains 248,320 logits; the complete normal-prefill capture
  file is byte-identical.
- The 288 normal-prefill continuation labels retain mean NLL
  **0.14113513363171917** and perplexity **1.1515802546926903** exactly.
  All three printed teacher-forced perplexities also match. These fixed
  fixtures measure regression consistency, not broad model capability.
- All four CPU contracts, eleven GPU operator tests and four model/state
  executables pass. Session checks include fresh-load replay, private batching
  and all 23 serving sampler strategies in AR/MTP. Rollback, rejection of
  version-19 snapshots by the version-20 build, and EOS checks pass.

[Identities, settings, results and reproduction](artifacts/qualification-snapshot-v20.json)
are retained compactly; raw runner records, command logs and captures remain
under ignored `artifacts/qualify/snapshot-v20-20261018/`. The runner's
cross-checkout command/test lookup was fixed and fixture-tested during this
exercise; the engine builds themselves are clean. Actual compiler, dependency
and runtime identities match independently of the Windows diagnostic
fingerprint's default compiler/driver labels. Linux, other quantizations,
vision/HTTP/parser contracts, original-checkpoint/conversion accuracy and
performance were not qualified by this exercise.

### Windows 4096-token internal chunk default

October 15, 2026: on top of the selected upstream prefill changes, raising the
internal limit from 2048 to 4096 retains all **648 complete vocabulary rows**
and both whole capture files, continuation NLL and perplexity exactly for
UD-IQ4_XS and UD-Q4_K_XL. The unchanged diagnostic covers three fixtures,
AR/MTP and prefixes through 32K. Both full session suites pass chunk splits,
private C2/C4/C6/C8 state, snapshot recovery, fresh loads, sampled p/q rejection
and residual correction, plus 23 serving sampler configurations in AR/MTP.
Formatting and the compiled resource guard pass. 4096 is now the fork
default. All 46 paired IQ4_XS AR HTTP responses also match text and
counts at C1/C2/C4/C6/C8, seeded-sampled C1 and image C1, with zero prompt reuse
and 16 output tokens each. Linux, multi-session memory and broader serving
latency remain unqualified. [Measurements, identities and scope](../../plans/prefill-chunk-4096.md).

### Windows selected upstream prefill changes

October 15, 2026: selected #421/#571/#463 changes on native Windows/gfx1151,
TheRock 10.0.0/clang 23, UD-IQ4_XS and UD-Q4_K_XL with their shared Q8_0 MTP
sidecars. Both builds use the same extended diagnostic over three prose/code
fixtures and 96/1023/1024/2048/4096/32768-token prefixes in AR/MTP modes.
All **648 complete vocabulary rows (160,911,360 F32 values)** and both complete
capture files are byte-identical; continuation NLL and perplexity are exact.

Both formats pass full session, serialized snapshot and all 105-prefix rollback
checks, including private C2/C4/C6/C8 state/RNG, sampled rejection/residual
correction, fresh loads and 23 serving sampler strategies. Independent operators
retain their existing F64 limits. The original compiled resource contract and
all twelve dense F16 variants have zero scratch/register spills.

All 92 paired HTTP responses also retain text and token counts: greedy
C1/C2/C4/C6/C8, seeded-sampled C1 and a real 256-by-256 image with the BF16
projector, in AR and full-head MTP modes. Each request emits 128 tokens and
reuses zero prompt tokens. This is functional parity; HTTP latency is not
fully qualified by one paired campaign.

This is regression/state qualification, not new unquantized-teacher,
conversion or encoder parity. Linux CI and GPU timeline profiling are unrun.
[Commands, identities, results and limits](../../plans/prefill-small-pr421-571-463.md).

### Windows PR #518 local decode kernels

October 13, 2026: stages 0/1 on native Windows/gfx1151 with TheRock 10.0.0,
unchanged compiler/dependency versions, UD-IQ4_XS and UD-Q4_K_XL targets and
the shared Q8_0 MTP sidecar. The preserved original `ab4d98189a6d` and final
kernel build `76656a5befc1` retain **1,236 identical complete target-logit rows**:
696 fixed-corpus scalar/verification/p8 hashes and 540 byte-identical normal
prefill/continuation rows through 4096 tokens. Mean NLL and perplexity are exact
within each schedule/format. All **128 fixed MTP stage hashes**, including
16 complete vocabulary rows, also match; full/ragged catch-up through 2048,
isolated batching and image-ID controls pass. The Q4 independent scalar MTP
audit passes unchanged numerical limits.

All 15 CPU/GPU operators pass, including wave64 and native IQ paths. Both
formats pass session, persistent snapshot, 105-prefix rollback and EOS suites,
with graph replay, private C2/C4/C6/C8 state/RNG, sampled rejection/residual
correction, 23 serving sampler strategies, request budgets, cancellation and
cache replay. Rollback also passes with Windows tuning disabled. The compatible
BF16 projector and two 256×256 fixtures retain all 24 complete-vector hashes
and perplexities for IQ4_XS, including partial-image/cancelled and seeded replay.

Two baseline gaps remain: the original IQ4_XS independent MTP attention audit
has relative RMS **0.0093689** against **0.002**; the Q4 two-image MTP stage-2
cached/cold greedy/seeded decode assertion fails on both original and candidate.
All 20 Q4 image vectors captured before that assertion retain identical hashes
and perplexities, but full Q4 image replay remains unqualified. Assertions and
limits were not relaxed. These are fork-consistency checks, not new conversion
or encoder qualification. Linux execution and GPU timeline profiling are unrun.
[Commands, identities and qualification scope](../../plans/decode-kernels-pr518-stages01.md).

### Windows prefill spill regression

October 7, 2026: TheRock 10.0.0/clang 23, UD-IQ4_XS and shared Q8_0 MTP,
matched baseline/candidate for [PR #459](../../plans/prefill-spills-pr459.md).
Schedules `1:2,4,8:p8` retain **549 identical full raw-logit hashes** and
perplexity **1.343238 / 1.343238 / 1.361810**, respectively. Equality is across
builds within each schedule. The prefill/session test passes chunk boundaries,
graph/snapshot replay and isolation through 4096 tokens. Independent paired
Q4_K/Q5_K routed operators pass and their former 80-byte private cache is gone.
The IQ4_XS model is a control for this paired-cache fix. Additional UD-Q4_K_XL
validation has 47 Q4_K gate/up layer pairs and one Q5_K pair (layer 2),
with eligible Q5_1/Q8_0 down projections. The four-shard target and shared Q8_0
sidecar load successfully on this Windows host. The new explicit session capture
retains **270 byte-identical full-logit rows** in AR/MTP modes after
96/1023/1024/2048/4096-token prefixes of three fixtures and eight continuations.
All 240 labels retain mean NLL **0.17457870930144928** and perplexity
**1.1907444613614468**. The model's snapshot/graph replay, execution-mode isolation
and split-prefill checks through 4096 tokens also pass. This closes the paired
Q4/Q5 full-model execution-consistency gap; Linux and sampled rejection parity
remain unrun. The separate teacher-forcing diagnostic still has its eight-row
limit; the session capture records each normal prefill's frontier instead.
[Exact identities and captured row hashes](artifacts/prefill-spills-pr459-q4.json).

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only` or `--prefill-only` on the session test;
the snapshot test covers persistent image/text state. For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

For focused Windows rollback validation, build and run the explicit model test:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target qwen38_flash_next_rollback_test
.\build\gpu-test\qwen38_flash_next_rollback_test.exe --model "$MODEL" --mtp-model "$MTP"
```

It checks every kept prefix of verification widths 2–8, repeated shapes,
restoration of the starting snapshot, reset, and subsequent full trunk logits.
Keeping one row means retaining the target anchor and rejecting all drafts.
Run once with normal Windows defaults and once with
`$env:GUFO_PLATFORM_TUNING = "none"`; remove the variable afterward. This uses
teacher forcing without advancing the MTP head. The separate snapshot test
checks real speculative snapshots, including deferred rejection replay. Neither
test establishes quality against an independent model implementation.

The explicit `qwen38_flash_next_eos_test` keeps EOS enabled and checks physical
position, committed tokens, full frontier logits against target replay, sampled
residual correction, snapshots, and serving stops/cancellation/continuation.
The [October 3 Windows audit](../../plans/flash-next-eos.md) records the tested
IQ4_XS/Q8_0 artifacts, cases, results and remaining coverage limits.

## Incremental host checkpoints on Windows

October 6, 2026, IQ4_XS target, Q8_0 MTP and BF16 projector: all 675 matched-token
full-logit hashes match the unchanged fork baseline, with perplexity 21.579505
in each schedule. Incremental and full snapshot exports are byte-identical in
AR/MTP, including direct disk streaming, branch overwrite, reset, cancellation,
eviction and source destruction. C2/C4/C6/C8 sampling/logits/RNG and residual
replay pass with Windows defaults and disabled platform tuning; all 105 explicit
rollback-prefix continuations and the AR/MTP image-prefix checks pass.

HTTP passes 27 growth checks, 15 history-edit checks and two cancellation/resume
cases. All ten matched timing replies and token/draft counts agree. Existing
version-15 disk checkpoints load correctly, and the old executable restores the
final candidate's streamed checkpoints with identical continuation output.
These are fork consistency checks; Linux validation is still required.
[Commands, scope and evidence](../../plans/flash-next-host-snapshots.md).

## Benchmark method

September 22-23, 2026; one warmed sample per point, greedy, thinking off.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. C1/2/4/6/8 use the same d0 prompts; every session prefills before
measured tg128, with at most four prompt-tail tokens reevaluated. Rates sum
individual decode rates. Gufo d0/C1 agree within 0.4% with matching drafts/output.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
