# IQ routed-expert F16 prefill qualification

Baseline: `0817d23b697460626d9da2bcb815b1a14ac1922c`, native Windows
gfx1151, TheRock 10.0.0 / clang 23. This is an experimental implementation
record, not a claim that the new arithmetic is qualified.

Code candidate: `ec31961208defb3ec2a941c5d003129ab3e4ee1a` (separate IQ
support `42274ba3`, then paired gate/up `ec319612`). Q4_K_XL qualification
passes; IQ qualification fails the unchanged exact-logit gate. Both formats
pass all required Windows functional/operator/state checks. Keep this on the
experimental branch; no production promotion or tolerance change is implied.

## Investigation corrections

The original root report identifies the dispatch gap correctly: the identical
1,224 tensor names/shapes give 48 eligible Q4 layers and zero eligible IQ
layers at the baseline. Several supporting claims need correction:

- The IQ fallback already uses INT8 WMMA on gfx1151. This is a difference in
  staging, activation precision and tile reuse, not missing matrix-core use.
- `ExpertMatrixRows` is true for every prefill width, including one token;
  the 32-row cutoff applies outside prefill. The routed readers are
  `RoutedF16RowBytes` and the tile staging code, rather than generic
  `RowElement`/`RowBytes`.
- IQ expert storage is **55.432129 GiB**, versus **71.728516 GiB** for Q4.
  The report's 69.78 GiB IQ estimate used incorrect IQ3_S geometry in a
  temporary inventory helper: 110 bytes represent 256 weights, not 32.
  The nominal active-expert byte estimate remains 23% lower for IQ, but
  that alone does not explain MTP throughput.
- IQ4_NL down-projection K=640 already fits the existing 64-weight stages
  as pairs of 32-weight blocks; a new K tile is unnecessary.
- The original HTTP rows differed in context and MTP vocabulary settings,
  and streamed chunk counts are not completion-token counts. They establish
  neither matched latency nor matched decode throughput.

Official PR #570 was closed unmerged. The two pinned reference commits below
provide the narrower prefill implementation; the source branch's broader
decode/loading/history changes are excluded. Existing selected #571
adaptations do not mean its complete upstream commit is an ancestor.

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
  not candidate quality evidence.

## Stage 2: paired IQ gate/up

- Expanded routed operators pass in 83.38 seconds. IQ3_S/IQ4_XS paired
  64/128-token tiles match separate F16 projections under the existing
  signed-zero convention and replay byte-exactly.
- All 27 routed variants are scratch/spill-free. Resource counts for all
  23 stage-1 variants are unchanged. The guard requires all four new paired
  variants, and its ten fixtures pass.
- Windows remains the priority. The user has WSL available but considers
  Linux validation low priority; Linux checks are deferred, not passed.
- One initial baseline normal-prefill attempt was stopped after its visible
  output file stayed empty for several minutes; that attempt overlapped an
  operator run initially. Its incomplete record is retained. A zero visible
  file size is insufficient evidence of a stall: a later serial Q4 capture
  also showed zero until near completion, then passed in 197 seconds. No
  definitive cause is assigned to the stopped attempt.
- The isolated IQ retry and complete runner-managed IQ baseline retain the
  identical 324-row file SHA-256
  `943e7bc22df220ad1b48799d8117e00b8adbcca8c330961313155218293ce210`.
  All subsequent GPU/model jobs run serially.

## Full-model qualification

Native Windows/gfx1151, TheRock 10.0.0/clang 23, matched GGUF shards, shared
Q8_0 MTP sidecars, corpus, tokenizer, seed 459 and schedules `1:2,4,8:p8`.
The plan selects arithmetic, prefill and state, with no uncovered inference
source paths. Both records use distinct retained binaries and no control or
mismatch overrides. The runner marks the candidate checkout dirty solely
because the original root report is untracked; tracked source matches the
candidate commit.

Both formats pass fresh native builds, four CPU contracts, eleven GPU
operator tests and all four model/state executables. Coverage includes
prefill chunk boundaries and tiny tails, graph replay, private C2/C4/C6/C8
state/RNG, fresh loads, 105 rollback prefixes, snapshots, sampled rejection
and residual correction, 23 serving sampler strategies, EOS, budgets, stops,
cancellation and cache replay. Production resource checks cover all 70
required gfx1151 kernels with zero failures. Linux remains deferred.

| Matched check | UD-IQ4_XS | UD-Q4_K_XL |
| --- | --- | --- |
| Runner verdict | **FAIL**, exact numerical comparators differ | **PASS** |
| Teacher-forced full-logit rows | 675/675 differ | 675/675 exact |
| Normal-prefill full-logit rows | 324/324 differ, all finite | 324/324 exact |
| Normal-prefill top-1 matches to baseline | 314/324 | 324/324 |
| Normal-prefill perplexity | 1.2427840354 -> 1.2844440694 (**+3.3522%**) | 1.1515802547, exact |
| Decode/verify fixture perplexity | 21.579505 -> 21.428664 | 22.829414, exact |
| p8 fixture perplexity | 21.002209 -> 21.069965 | 21.998551, exact |

The 324-row normal-prefill fixture covers three repeated text/code patterns,
AR/MTP, six prefix lengths through 32K and eight teacher-forced continuations
after each prefix (288 labels). AR and MTP capture halves are byte-identical
within each candidate; they are not independent quality samples. IQ mean
KL(baseline || candidate) is 0.0280697, maximum 1.43667; maximum full-logit
RMSE is 1.45578. The largest changes occur in prose continuations after 2K
and 4K prefill. These are measured regression fixtures, not broad capability
or original-checkpoint accuracy scores.

All three IQ teacher-forced schedules begin with a prefill call, including
the initial one-token prefix, so changed hashes in the decode schedules do
not imply that decode kernels were edited. F16 prefill changes the starting
state subsequently consumed by the unchanged decode/verification paths.
Passing state checks does not accept this numerical transition. The current
declaration has no model-owned numerical limits for it; none were invented
or relaxed after seeing these results.

## Independent CPU reference

Three fixed prose/code prompts, batch 64 and context 4096, compare all 52
GPU prompt-prefill vocabulary rows with the unchanged CPU `ReferenceModel`
using the same quantized GGUF and token histories. Recomputed CPU top-1 IDs
are unchanged across the baseline and candidate builds.

| GPU versus independent CPU formulas | Baseline | Candidate |
| --- | ---: | ---: |
| Top-1 agreement | 51/52 | 52/52 |
| Mean KL(CPU || GPU) | 0.00439308 | 0.00250885 |
| Maximum KL | 0.03179 | 0.03374 |

The probe prints KL to five decimals. Mean agreement improves, while its
worst row worsens slightly. This is an independent short-prompt arithmetic
check against the quantized checkpoint, not an original unquantized teacher
or a substitute for a justified deep-context numerical policy. It does not
resolve the measured repetitive-continuation perplexity increase.

## Fixture perplexity attribution

October 19 follow-up from the retained captures; no new model run. Per-label
NLL over the 324-row capture, with labels reconstructed from the three
repeated token patterns, reproduces the recorded baseline, candidate and Q4
mean NLLs to 1e-15. The fixture repeats a pattern tokenized once, so every
wrap places a string-initial `Virtual` or `A` (no leading space) after `". "`.
The model expects ` Virtual`, ` A` or `<|im_end|>` there; those labels carry
high, unstable NLL in every build.

| AR labels (MTP rows duplicate them) | Labels | Baseline | Candidate | Q4_K_XL | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 144 | 1.24278 | 1.28444 | 1.15158 | +3.35% |
| Pattern wrap (space or string-initial token) | 22 | 3.87579 | 4.88280 | 2.42342 | +25.98% |
| Interior | 122 | 1.01233 | 1.00956 | 1.00699 | **-0.27%** |

Three wrap positions, doubled by the identical AR/MTP halves, contribute
+13.4 nats against a net +9.5 nats; 230 of 288 labels improve. Among the
wrap labels 14 improve and 8 worsen. The increase is a tokenization-boundary
artifact of this fixture, not a broad shift.

Weight rounding is excluded independently. Emulating the kernel's
`fma_f16(q, f16(scale), 0)` on every 32nd expert of every layer (3.7 billion
real weights) against exact F32 dequantization gives median row relative
error 2.98e-4 for IQ3_S, 2.77e-4 for IQ4_XS and 2.07e-4 for IQ4_NL, against
2.06e-4 for the already-routed Q8_0 down layers. Seven of a billion IQ4_NL
weights are F16 subnormal; none flush to zero. Kernel-variant dependence is
also excluded: the candidate's session suite already requires byte-exact
logits for a 2048-token prefill split at 1025, which mixes the separate and
paired gate/up kernels and different down tile maps.

## Natural-text check

Wikitext-2 test, first 109,682 bytes cut at a line boundary (SHA-256
`ea9139f01dd970bf507de446176c2bf27f22c6229659fa611dab0a3888d19e29`;
full file `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
26,574 tokens and 26,572 scored positions. Both retained release binaries
run `gufo bench --logit-eval` with the shared Q8_0 MTP sidecar and prefill
schedules `p8:p5`. Prefill logit rows are capped at draft tokens plus one,
so p8 is the widest scored prefill chunk. Prefill always takes the routed
path, so p8 exercises the candidate's F16 experts. Intervals are 95%
block-bootstrap (256-position blocks).

| Comparison | Perplexity | Change | 95% interval | Mean top-64 KL | Top-1 agreement |
| --- | --- | ---: | --- | ---: | ---: |
| Baseline -> candidate, p8 | 2.534419 -> 2.540233 | +0.229% | -0.103% .. +0.594% | 0.0182 | 96.08% |
| Baseline -> candidate, p5 | 2.539890 -> 2.538083 | -0.071% | -0.330% .. +0.232% | 0.0182 | 95.89% |
| Control: baseline p8 -> p5 | 2.534419 -> 2.539890 | +0.216% | -0.054% .. +0.457% | 0.0139 | 96.69% |
| Control: candidate p8 -> p5 | 2.540233 -> 2.538083 | -0.085% | -0.337% .. +0.171% | 0.0109 | 97.08% |

The build change has no measurable natural-text effect: its sign follows the
chunk width and is the same size as re-chunking either build. Below position
2048, where both builds are chunk-width exact, the candidate improves
perplexity from 1.769443 to 1.763374 (-0.34%, interval -1.76% .. +0.96%)
and top-1 accuracy from 85.05% to 85.64%; 1,296 positions improve and 751
worsen.

Both builds are byte-exact between p8 and p5 for positions below 2048, then
differ on 92% of later rows. That is where `indexer_top_k` = 2048 sparse
attention starts selecting 4-token compressed blocks. The fixture's largest
changes, in prose continuations after 2K and 4K prefill, sit in this regime.
The width dependence predates this candidate and has three causes:

- The sparse WMMA kernel packs four queries counted from the chunk start and
  loaded values past the group's last query whenever the batch continued.
  Zero probabilities times those values changed the PV sum's last bits. The
  kernel now reads every key past the group as zero padding, so a group's
  result no longer depends on where the batch ends. 4-aligned prefill, and so
  the qualification capture, is byte-identical; p8 now equals p4 exactly on a
  2,579-token wikitext slice.
- Unaligned chunk starts still change which queries share a key union.
- The indexer's BF16 key and query projections run through hipBLASLt, whose
  kernel and edge tiles depend on the batch width. The deterministic F16
  route for the router and alpha/beta projections does not cover BF16.

On that slice, the 530 sparse positions changed top-1 on 14 rows between
p8 and p1 before the kernel change and on 9 after it.

This is a quantized-checkpoint comparison over one natural-text slice, not
an unquantized teacher or capability score, and it does not supply the
model-owned numerical limits the qualification gate requires. Dumps and the
comparison scripts are under ignored `build/iq-ppl-cause/`.

## Exploratory production timings

These measurements do not override IQ's failed exact numerical gate and are
not promoted into `BENCHMARKS.md`. Native release binaries use the same
TheRock/clang and model artifacts as the qualification records. Firefox's
earlier roughly 17 GiB GPU reservation is absent; each timing arm records
the remaining background process reservations separately.

CLI: AR, depth zero, pp2048/pp4096/tg128, context capacity 4097, temperature
zero, seed 459, no speculation or MTP sidecar. Each model runs ABBA with a
fresh process per arm, warmed benchmark shapes and three timed repetitions
per shape. Values below average the two arm means per build; model loading
is excluded. Raw logs retain each arm's printed standard deviation.

| Model / workload | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| IQ pp2048 | 989.620 | 1544.525 | **+56.07%** |
| IQ pp4096 | 1154.835 | 1608.825 | **+39.31%** |
| IQ tg128 | 28.285 | 28.345 | +0.21% |
| Q4 pp2048 | 1600.365 | 1593.290 | -0.44% |
| Q4 pp4096 | 1647.695 | 1645.835 | -0.11% |
| Q4 tg128 | 28.630 | 28.595 | -0.12% |

The IQ pp2048 arm means are 985.64/993.60 baseline and 1544.33/1544.72
candidate. Q4's small changes and IQ decode movement do not establish
meaningful gains or regressions. IQ now runs within about 3% of the Q4
candidate at pp2048 and 2% at pp4096. This closes most of the measured
prefill gap without a persistent expanded expert copy or decode changes.

HTTP: separate fresh C1 server per ABBA arm, context 8192, shared Q8_0 MTP,
full draft vocabulary, seven-draft ceiling, greedy seed 459, thinking disabled.
Every request processes the same 1504-token prompt and emits 16 actual tokens;
terminal usage and timings both confirm zero cached prompt tokens. The
retained JSON includes a reconstructable prompt and its SHA-256.

| Model / scope | Baseline mean | Candidate mean | Change |
| --- | ---: | ---: | ---: |
| IQ cold TTFT | 2.428 s | 1.998 s | **-17.71%** |
| IQ complete request | 2.957 s | 2.579 s | -12.80% |
| IQ launch to ready | 22.562 s | 22.295 s | -1.18% |
| Q4 cold TTFT | 1.779 s | 1.787 s | +0.45% |
| Q4 complete request | 2.309 s | 2.307 s | -0.05% |
| Q4 launch to ready | 26.097 s | 25.746 s | -1.34% |

IQ TTFT arms are 2.452/2.404 s baseline and 2.162/1.835 s
candidate. Both candidate arms improve, but this short cold-request campaign
does not establish a general serving-latency guarantee. IQ's generated text
differs between builds and repeats within each build; complete-request and
decode times therefore do not compare identical output work. Q4 text hashes
match across all four arms. Startup changes are small and unqualified.
Background process reservations span only 5.6825-5.6935 GiB across all 16
CLI/HTTP arms, led by DWM at roughly 4.74 GiB. These Windows counters report
reserved dedicated bytes, not physical residency. No other model/GPU tests
run during timing, and no benchmark-owned process remains running afterward.

## Reproduction and decision

[Compact identities, checks, numerical comparisons and timing arms](../models/qwen3.8-flash-next/artifacts/iq-prefill-f16-candidate.json)
retain full binary/model/sidecar/corpus hashes and actual toolchain identity.
The diagnostic fingerprint's compiler/driver strings are Windows defaults;
they are not substituted for the separately recorded TheRock/clang identity.
Full logs, raw logits and command records remain under ignored
`artifacts/iq-prefill/`; build/operator logs are in `build/iq-prefill/`.
Build and record from checkouts at the pinned baseline and code-candidate
commits above; a later documentation commit is not this plan's head revision.

```powershell
python tools/models/qualify.py plan --base 0817d23b697460626d9da2bcb815b1a14ac1922c --head ec31961208defb3ec2a941c5d003129ab3e4ee1a --format json --output artifacts/iq-prefill/plan.json
# Run serially for IQ and Q4, with the corresponding first shard and sidecar.
python tools/models/qualify.py record --model qwen3.8-flash-next --side baseline --plan artifacts/iq-prefill/plan.json --repo "$BASE_REPO" --gufo "$BASE_GUFO" --tool "session_test=$BASE_SESSION_TEST" --file "gguf=$MODEL" --file "mtp=$MTP" --out artifacts/iq-prefill/baseline
python tools/models/qualify.py record --model qwen3.8-flash-next --side candidate --plan artifacts/iq-prefill/plan.json --repo . --gufo build/release/gufo.exe --tool session_test=build/gpu-test/qwen38_flash_next_session_test.exe --file "gguf=$MODEL" --file "mtp=$MTP" --env "VCPKG_ROOT=$VCPKG" --out artifacts/iq-prefill/candidate
python tools/models/qualify.py compare --model qwen3.8-flash-next --plan artifacts/iq-prefill/plan.json --baseline artifacts/iq-prefill/baseline --candidate artifacts/iq-prefill/candidate --out artifacts/iq-prefill/comparison
# Each release arm, with baseline/candidate binary selected in ABBA order:
& $GUFO bench --model "$MODEL" --speculative off -p 2048,4096 -n 128 -d 0 -r 3 --temperature 0 --seed 459
& $PROBE --model "$IQ_MODEL" --batch 64 --context 4096 --prompt "$FIXTURE" --reference --dump "$LOGITS"
# Fresh HTTP server per arm; wait for matching /props identity before requesting:
& $GUFO serve --host 127.0.0.1 --port $PORT --sessions 1 llm --model "$MODEL" --context 8192 --mtp-model "$MTP" --speculative mtp --draft-tokens 7 --mtp-draft-vocab full --served-model-name gufo
```

The narrow staging adaptation is an effective prefill experiment: large
native gains, a smaller cold-TTFT gain, no persistent expert expansion and
no meaningful Q4/decode control movement. **Do not promote it as qualified.**
The independent short-prompt reference improves on average. The repetitive
fixture's increase is confined to tokenization-boundary labels, and wikitext
shows no effect beyond chunk-width noise. None of these supplies a
model-owned acceptance policy. Promotion needs declared numerical limits for
this F16 transition; the alternative is to require the old INT8 arithmetic
and pursue a bit-preserving optimization. Existing exact,
operator and state gates must remain intact. Linux, multi-client timing,
GPU timeline profiling and broader teacher/capability evaluation are unrun.
The timings cover the combined F16/paired candidate; a full-model pairing
ablation is also unrun.
