# Windows upstream sync log

Track merges from [official Gufo](https://github.com/gufo-org/gufo) into the
Windows integration branch. This log starts after the October 2026 forward-port;
it is not a retroactive inventory of the old fork's selective imports.
An imported PR is not automatically qualified on Windows.

## Windows acceptance policy

This is a Windows-specific Gufo port. Preserve Windows compatibility,
correctness, stability and performance when accepting upstream changes.
Upstream inclusion or passing Linux checks does not override a Windows
regression.

- Check the known-regression entries before every sync. Skip or defer the
  code from PRs with confirmed Windows regressions until a fix passes the
  affected Windows checks. A focused compatibility fix may be retained with
  the PR when it is validated and documented.
- Preserve official ancestry: normally merge the recorded upstream head,
  then explicitly revert the regressing PR on the candidate. Record it as
  **reverted/deferred**, rather than imported and validated. If dependencies
  prevent a safe isolated revert, defer the dependent changes or the candidate
  until the compatible combination is validated.
- Confirm attribution against the previous Windows base with matched inputs
  and toolchain. Keep inherited failures, missing coverage and noisy results
  visibly pending; they are not automatically evidence that an incoming PR
  caused a regression.
- Record every skipped, reverted or deferred PR in the sync entry and known
  regressions section. Include its URL and merge SHA, affected Windows behavior,
  reproduction/toolchain details, baseline and candidate evidence, disposition
  and revert/fix SHA, affected dependencies, and required reintroduction checks.
  Keep failed observations and the original entry when a later fix resolves it.
- Preserve these exclusions during future conflict resolution, backports and
  related upstream changes. Reapply a deferred PR only after reviewing its fix
  and passing the recorded Windows checks against the previous Windows base.
  Append the fixing SHA, validation evidence and new status to its record.

## Known upstream regressions

[`known-regressions.json`](../tools/windows/known-regressions.json) mirrors
these entries for the sync tools: PR, merge/revert SHAs, identifiers that must
stay absent while reverted, and reintroduction checks. Update both together.

### PR #688: Windows DFlash2 continuation latency

**Bounded pinned prompt capture is deferred on Windows.** Official
[#688](https://github.com/gufo-org/gufo/pull/688),
`b0eb5d476d07b659fb0d6a448b56f66d5420a113`, is an ancestor through
merge `225ec080`; explicit revert `432163e9` removes the optimization.
Commit `77a58de7` separately retains only its upload-failure cleanup and
unchanged upstream lifetime fixture. Successful injections keep the existing
stream wait; failed injections now drain outstanding readers before returning.

Two candidate runs (`792150d6` binary SHA-256 prefix) and three unchanged
baseline runs (`b2df4482`) use the same executable path, all 143 runtime files,
Qwen 27B Q4_K_XL/official Q4_K_M DFlash2/projector, 32K context, two sessions
and effective 4 GiB RAM caches with equal staging/disk capacities. In the
first `long-context` suite, before the later divergent cancellation histories,
all outputs, prompt/cache work and sampled replay drafts match: 12,454 prompt
tokens reused in full, seven completions, 11 proposed/five accepted drafts.

| Matched sampled replay phase | Baseline range (ms) | Pinned-capture candidate range (ms) |
| --- | ---: | ---: |
| Client wall | 351.75-358.76 | 405.42-419.80 |
| Cache restore | 15.27-16.61 | 20.77-21.44 |
| Server TTFT | 19.40-21.70 | 26.30-26.70 |

The unchanged upstream timing qualification finds **nine failed, 73 passed and
42 inconclusive cells** for this entire first suite at the existing 5% and
3 ms limits. The scoped evidence copies every first-suite request byte for
byte, verifies settings/budgets and retains source hashes; it makes no claim
about later histories. The full comparisons and their failures remain intact.
No inference workaround or timing-margin change is retained. Attribution is
to the tested #688 combination; the complete runtime cause is not established.

The full DFlash2 comparison also fails on an image continuation after different
cancellation frontiers (273 versus 278 cached tokens) change seeded output.
Both builds replay exactly within themselves. Explicit matched-frontier image
probes produce identical output/work across builds and reproduce both original
output hashes on baseline, so this is not evidence of new transfer corruption.
An unchanged baseline also reproduces the legacy-tool seven-token re-prefill
variation. Neither observation is relabelled as a full comparison pass.

A delayed-copy probe confirms Windows pageable D2H copies already block while
pinned copies return asynchronously. Omitting the upstream waits would therefore
be speculative; no such patch was made. Physical-memory pressure, other HIP
failures and a full performance qualification remain open.

Evidence ([evidence archive](#evidence-archive)): `sync-688-2026-10-23/`,
especially `long-context-qualification.json`, `qualify-long-context.py`, original/repeat
functional reports, `fixed-image-frontier/` and `copy-semantics.log`.
Reintroduction must preserve upload draining, pass affected numerical and
cancel/image/cache/batch checks, verify shared-cache Flash-Next AR/MTP behavior,
and clear matched per-request timing gates with unchanged-base controls.
Also run varied 10-16K-token first requests with reuse on/off and pp2048/tg128.
The revert remains effective in future merges until deliberately qualified.

### PR #699: Windows first-request MTP decode regression

**Avoid reapplying #699 unchanged on Windows.** Its code is reverted by
`161035d8`; upstream ancestry is retained by merge `54fb4b4e`. PRs #692,
#540 and #690 remain included. This is a Windows/TheRock observation, not
an assertion about Linux performance.

With the same model, toolchain, runtime DLLs, server path and 11,036-token
fresh-process request, pre-sync first MTP decode measures 57.51, 53.86 and
51.32 ms in separate controls. The full import measures 60.90 and 63.35 ms;
removing #699 alone measures 53.55 ms. The rebuilt explicit revert measures
53.25/51.52 ms on first/repeat decode, with identical output and cache/draft
work. Its 6,557.75/6,284.98 ms prefill also remains within the baseline gate.
The focused import/base pair exceeds the unchanged 5% and 3 ms request-phase gate. Warm repeats are about 51-53 ms;
outputs and prompt/cache/draft counts agree. See the model's
[individual observations](models/qwen3.8-flash-next/WINDOWS.md#sync-first-decode-control-october-21-2026).

Temporary host-stage traces locate roughly 7-9 ms in the first neural
predictor's Q8 MoE enqueue after KV-only prefill. The baseline pays this
first-use cost during prefill instead. This suggests lazy HIP setup moving
into decode; it does not establish a complete root cause. Removing eager
vision setup does not fix it (59.60 ms). Restoring full scalar predictor
catch-up measures 52.75 ms once, but the combined scalar/batch fallback
measures 66.46 ms. Explicit startup function preparation also fails (67.02 ms).
None of those experimental changes is retained. Profiling measurements are
diagnostic and are not qualification timings.

Reported upstream as [gufo-org/gufo#745](https://github.com/gufo-org/gufo/issues/745).
Official #733 (lone greedy MTP requests use the small draft head) does not
fix it. On the October 23 Windows tree plus #733, ABBA fresh-process controls
measure first decode at 62.47, 61.35 and 64.56 ms with #699 versus 52.93, 51.45
and 53.58 ms without it; repeat decode, prefill, output and cache/draft counts
are unchanged. #733 alone is neutral. Evidence
([evidence archive](#evidence-archive)): `exp-733-2026-10-23/`. #733 itself is
not imported yet.

**Reintroduction requirement:** review the upstream fix, then run
[the first-request control](#reproduce-the-first-request-control) against the
previous Windows binary. Retain every
phase, output and cache/draft count; a warmed benchmark alone is insufficient.
A later merge will not automatically undo the revert because #699 is already
an ancestor. Any deliberate reapplication must cite this entry and pass the
focused cold-start control plus affected correctness checks.

## Evidence archive

Sync evidence is written to ignored `build/sync-*` directories in the task
worktree. Before that worktree is removed, its text artifacts (JSON, JSONL,
logs, scripts, Markdown, diffs) are copied to the maintainer's evidence
archive under the same directory name; on the reference host that is
`C:\P2027\gufo-evidence\`. Evidence references in this log name those
archived directories. Binaries, model files and caches are not archived.

## Batching

Import official history in batches rather than one PR at a time. A batch must
end at an official commit so that the merge keeps official ancestry. It is
therefore a contiguous run of incoming commits; selecting non-adjacent PRs
would mean cherry-picking. Only the cut points are chosen.

- **Cut at release tags by default.** A `vX.Y.Z` tag passed upstream's
  complete Devlab functional suite, and merging its release commit keeps
  `version.txt` and `CHANGELOG.md` in step with official. Take the inventory
  from the commit range, not the release notes: notes omit `build`, `ci`,
  `test` and `chore` PRs and have missed late-merged `feat`/`perf` PRs
  (v0.22.0 omits #705; v0.23.0 omits #752).
- **Classify every incoming commit by what it requires**, using the path rules
  in [`sync-batching.json`](../tools/windows/sync-batching.json):

  | Class | Meaning | Required work |
  | --- | --- | --- |
  | convergence | Windows portability areas: `compat/`, `src/core/platform/`, `src/core/hip/`, `src/cache/`, Windows workflows | Choose the fork or upstream implementation for each overlap, review the affected contracts, run the Windows CPU gate and HIP build |
  | timing | Hot paths of models with Windows timing evidence, shared decoding code, and source reverted for a known regression | Matched Windows timing controls and the first-request control |
  | contained | Other source and tests | Windows CPU gate and affected functional suites |
  | inert | Documentation, Nix, CI, functional suites, release commits | Documentation, formatting and build checks |
  | shared (flag) | Touches a source or test file listed in `shared-edits.json` | Review the fork's edit and its contract, even after a clean merge |

- **End a batch early** when the next commit would exceed two convergence or
  four timing commits. Keep related convergence work, for example one
  platform leaf and its adoption, in one batch rather than mixing several
  unrelated fork-or-upstream decisions into one merge. **Continue past a
  release** whose batch has no convergence or timing commit.
- **Merge each batch end as its own checkpoint** on a single candidate branch,
  then run that batch's checks before merging the next. One sync PR and one
  log entry cover all checkpoints; record each checkpoint's official SHA and
  merge commit in the entry's checkpoint table.
- **Narrow failures by class.** When a batch fails a timing gate, revert-test
  only its timing commits, listed as timing suspects; when the CPU gate fails,
  bisect the batch's first-parent commits. Then apply the acceptance policy to
  the responsible PR.

The inventory applies these rules under **Suggested batches** and lists each
batch's merge target, per-commit classes, required checks and timing suspects.
The suggestion is advisory: move a cut point when review shows a dependency
or a lower-risk boundary, and record the reason in the sync entry.

## Repeatable procedure

1. Follow the [sync skill](../.agents/skills/sync-windows-upstream/SKILL.md).
   Verify [repository roles](WINDOWS-COMPATIBILITY.md#repository-roles),
   worktree status and local versus remote `windows-main` before fetching. Preserve unpublished Windows work.
2. Record the Windows base and fetched `official/main` SHA before merging.
   Run `tools/windows/sync-inventory.ps1 -WindowsBase WINDOWS_BASE
   -OfficialHead OFFICIAL_HEAD -OutFile build/sync-YYYY-MM-DD/inventory.md`.
   It lists **all** incoming commits, resolves PR titles and merge SHAs with
   `gh`, lists release commits separately, reports files changed on both sides
   and upstream work on reverted regions, suggests [batches](#batching) and
   emits an entry skeleton. Exit code 1 means the sync imports a known
   regression.
3. Apply the Windows acceptance policy and known-regression entries above.
   Choose the batches. On an isolated candidate descended from the Windows
   base, merge each batch's exact official SHA in order. An already isolated
   T3 task branch is a suitable candidate. Preserve official ancestry through
   a merge commit per checkpoint. Review cleanly merged Windows call sites as
   well as conflicts; retain both platforms' evidence.
4. Review every [compatibility contract](WINDOWS-COMPATIBILITY.md) at each
   checkpoint. Audit new callers and moved implementations. Run each batch's
   required checks before merging the next one, then the bounded Windows CPU
   gate, documentation/formatting checks and the available Linux gate on the
   final candidate. Preserve individual timing results and failures.
   Revert/defer confirmed Windows regressions and validate the resulting
   combination before accepting the sync. Rerun the inventory with
   `-Candidate HEAD` (it fails if reverted code is present), run
   `tools/windows/diff-guard.ps1` (see the
   [maintained difference](WINDOWS-COMPATIBILITY.md#maintained-difference)),
   and open a draft PR against `windows-main`:
   pushing a `sync/official-*` branch alone triggers no hosted checks. Watch
   the Linux and Windows CPU runs with `gh run watch`.
5. Append an entry below with the PR table, base/head identities, resolutions,
   per-contract outcomes, exact checks, evidence location and pending limits.
   Put model measurements in the model's `WINDOWS.md` sidecar, not in the
   upstream model documents.
   List skipped/reverted/deferred PRs and their reintroduction requirements
   explicitly; do not silently omit them from the inventory. Record new
   reverts in `known-regressions.json`.
   Verify each PR merge SHA and the recorded official head with
   `git merge-base --is-ancestor SHA HEAD`. Do not replace old entries.
6. Integrate a qualified candidate into `windows-main` only after rechecking
   its base. Publication follows the requested endpoint and the
   [merge policy](WINDOWS-COMPATIBILITY.md#repository-roles).

For the next sync, copy the shape of the latest entry and replace its evidence.
For a batched sync, keep the skeleton's checkpoint table and give each
checkpoint its own merge review and validation subsections.
Use **validated**, **reviewed/unchanged**, or **pending (reason)** per contract;
do not count missing models, excluded disk persistence or noisy timings as passes.

For Windows functional timing comparisons, use the budget preflight wrapper:

```powershell
python tools/windows/compare-functional.py --pair BASE_RUN CANDIDATE_RUN --output NEW_COMPARISON.json
```

It delegates to the upstream functional comparator with the same per-request
gates and accepts its repeated `--pair` and `--control` arguments. Before
comparing timings, it reads both server startup logs in every run and rejects
clamped explicit RAM requests or different effective RAM, staging or disk
capacities. Identical launch arguments alone do not establish matched memory
budgets. Keep rejected runs as evidence; do not use them to clear earlier
matched failures.

### Reproduce the first-request control

Finish builds first and stop nothing you do not own. Run
[`first-prompt-control.py`](../tools/windows/first-prompt-control.py):

```powershell
python tools/windows/first-prompt-control.py --base-binary PREVIOUS\gufo.exe `
  --head-binary CANDIDATE\gufo.exe --repeats 2 --output build/sync-DATE/first-prompt
```

For each mode (AR with `--speculative off`, MTP with the shared-Q8 sidecar),
it alternates base/head in ABBA order. Each arm stages the executable, its DLLs
and `hipblaslt/` at the same path, then starts a fresh private server with
`--context 32768 --sessions 2 --think off --cache-ram-bytes 8589934592`. It
POSTs the retained 11,000-word body below to `/v1/chat/completions`, first
with `cache_prompt: true` and then with `cache_prompt: false`, and stops the
server with a console break. Models default to `Models` in
`tools/windows/local.json`. Omit `--head-binary` for an unchanged-binary
control.

The output directory must be new. It keeps `control.json` with settings,
binary/DLL/model identities, commands, `/props`, complete responses and
wall/phase timings. It also keeps one server log per arm and
`comparison.json`. [`compare-phases.py`](../tools/windows/compare-phases.py)
reproduces the comparison from `control.json`. Correctness problems fail the
run: wrong prompt reuse, a wrong answer, an HTTP error or an unclean stop.
Each phase is compared on its own with the repository's 5% and 3 ms rule; it
counts only when every repeat moves the same way, and it is compared only when
both arms did the same prompt/completion/draft work. Exit status 0 means
passed, 1 failed and 2 inconclusive. An unchanged-binary control with any
moved phase is inconclusive and measures noise. Investigate flags with more
repeats or a focused control, and save failed observations as well as
successes. For these artifacts, expect 11,036 prompt tokens, zero reused
tokens and `ACK`.

On this host, the first GPU submission after the device idles for about 5 s
stalls for 0.3–0.9 s in the driver. This happens in AR and MTP, with any
binary, and also mid-process after an idle gap. A fresh arm's first request
hits it at random (7 of 16 short MTP arms, October 22, 2026), inside slot
reassignment. A settle delay makes it certain rather than removing it, and a
warm-up request would defeat the control. The comparison therefore derives
each request's `slot_wait_ms` from the server's cumulative
`component_cache.global_spill.residual_wait_ms`, subtracts it from the wall,
queue and TTFT phases that contain it, and reports `slot_wait_ms` without
gating on it. A slot wait that is slower in every repeat gets a note: check the
server logs before calling it noise. Evidence that lacks the field is compared
unadjusted.

The retained request body, for reference:

```python
import random

words = ("river town family valley stone bridge boats timber salt wool market "
         "wheat children forest quiet garden road lake tree green water blue").split()
passage = " ".join(random.Random(813).choices(words, k=11000))
body = {
    "model": "gufo",
    "messages": [
        {"role": "system", "content": "Read the archive. Follow the final user instruction exactly."},
        {"role": "user", "content": passage + "
Reply with exactly the word ACK."},
    ],
    "max_tokens": 128, "temperature": 0, "seed": 42,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

## October 23, 2026: PR #677

Status: **merged** by PR #79 as `ad3696f5`. Local Windows checks passed, and
all six executed hosted gates passed on `bc6f8c6e` (three GPU jobs skipped).

- Windows base: `041772cbbca43953b5ba6b5322a18118ffe5bdba` (#78 merged).
- Previous official ancestor: `b0eb5d476d07b659fb0d6a448b56f66d5420a113`.
- Imported official head: `8a79230617490ff374b38edc002a8e68d241cf68`.
- Integration merge: `07edb498148c9db65213120aa9a967122cff0536`.
- Scope: #677 and its predecessor, the official 0.21.0 release commit
  `7110c795` (changelog, `version.txt` and release manifest, imported unchanged).
  #711 and later commits are outside this layer.
- Evidence ([evidence archive](#evidence-archive)): `sync-677-2026-10-23/`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#677](https://github.com/gufo-org/gufo/pull/677) | fix(cache): free RAM before writing oversized checkpoints to disk | `8a79230617490ff374b38edc002a8e68d241cf68` |

### Merge review

One conflict in `ComponentTextCache::Capture()`. Upstream's new over-RAM
branch compares against `cache.budget`, which exists only with the deferred
[#688 pinned budget](#pr-688-windows-dflash2-continuation-latency). The
resolution keeps #677's block, including the `refused`/`failed` logging and the
rethrow, and keeps the fork's `cache.ledger.Limits().ram_bytes` limit, as the
neighbouring snapshot reservation and planned-capture checks already do. No
#688 identifier returns.

Clean merges reviewed: deferred, largest-first staging allocation in
`CaptureDurable` with reclamation that protects the source checkpoint and this
turn's edit boundaries; the retention victim guard; the fake adapter's
`rows_per_chunk`; and the new retention and component tests. The new component
test's `/tmp` `mkdtemp` template already resolves through the Windows shim. The
functional README keeps the Windows and #699 paragraphs and adds the reclaim
case. The two inventory review items (#688 and #699 shared files) are covered
above; neither reintroduces reverted code.

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Reviewed/unchanged: no model I/O or n-gram change. |
| WDDM admission budgets | Reviewed/unchanged: device budgets are untouched. The over-RAM path is bounded by the host RAM ledger, verified at a 256 MiB cache below. |
| Pinned memory and transfer lifetime | Validated: over-RAM capture borrows live rows and stages private state; a 1.6 GB checkpoint publishes and restores with identical output across base/head. |
| Sockets and shutdown | Native CPU checks pass; every live server stopped cleanly. |
| Filesystem identity, privacy and failures | Validated: Windows NTFS rebuildable store (`disk_cache_platform platform=windows`) publishes, restores and restarts; native CPU store checks pass. |
| GPU queue policy | Reviewed/unchanged. |

- Native Windows CPU build and **16/16 CTests pass**, including
  `cache_retention_test` with the new protected-source case.
- `component_text_cache_test` still stops at the known, pre-existing
  `OrphansLeaveBeforeCheckpoints` DACL assertion (#688 entry) before #677's
  cases. An ignored harness compiled the unchanged upstream test with only
  `OverRamCaptureStagesPrivateStateUnderPressure(false/true)`; both pass.
- HIP `gufo` builds at `07edb498`; SHA-256
  `5929ab89ac4c59193b7861de93b56d49581eb5b4af58e87ea59cc9c8abced89b`.
  Base is the #78 production binary `c8e66f75...`.
- Native form of upstream's `publication_wait.py --case reclaim` (the upstream
  runner needs the Linux `LD_PRELOAD` disk gate, so it waits for the real
  publication instead): Flash-Next Q4_K_XL/shared-Q8 MTP, context 65,536, one
  session, 256 MiB RAM cache, 55,111-token deep prompt.
  - Base reproduces the bug: the deep capture is refused
    (`reason=byte_capacity`), nothing over RAM publishes and the retry
    re-prefills 55,073 tokens (TTFT 33.0 s).
  - Head reclaims a retained RAM checkpoint (`action=2 reason=1`), publishes
    the 55,106-token, 1,646,239,144-byte checkpoint and restores it from disk
    with 19 prefill tokens (TTFT 1.50 s). No `capture_failed`.
- At a 1 GiB cache (no reclaim needed) base and head both publish and restore.
  The first head/base pair differed by +166 ms in `cache_restore_ms` only;
  two further alternating pairs measured +0.8%/+2.7% restore and +0.9%/+3.0%
  TTFT, within the 5% limit. Restore timing drifts upward across the session
  for both binaries.
- Output: seed, deep, retry and cold responses and their draft counts are
  identical across all eight base/head runs. The retry's reasoning text differs
  from the cold control's in every run, including base retries that never touch
  disk; the final answer matches. This pre-existing prefill-split difference is
  why the upstream runner's exact retry/cold assertion is not a Windows gate here.
- Bounded MTP `cache_startup.py --case ram` and `--case disk` pass on head:
  2,294-token cold prompts, zero-prefill retries and a 7-token-prefill disk
  restart restore, with matching output and 19/16 drafts each. These runs set
  `PYTHONUTF8=1` because upstream harness provenance reads non-ASCII test files
  with the locale codec.
- Formatting checks 622 C++ files. Inventory with `-Candidate HEAD` and the
  maintained-difference guard pass; #688/#699 identifiers stay absent.
- Pending: upstream `publication_wait.py --case reclaim` itself (Linux-only
  fault library), Linux/hosted gates, and Qwen 27B AR/DFlash2 reclaim. No
  full model sweep or speedup claim.

## October 23, 2026: PR #688, pinned capture deferred

- Windows base: `d812ebab17f604684fef3513eb2d31560577d669` (#77 merged).
- Previous official ancestor: `3aa068b419e6272482298c1c6daccafb122764fa`.
- Selected official head: `b0eb5d476d07b659fb0d6a448b56f66d5420a113` (#688 only).
- Fetched official/main: `de43b8a047735fb296a5d4f1218f1f628cbff5ac`; later changes
  are outside this layer. No release/version changes.
- Conflict-free integration merge: `225ec080`; explicit optimization revert:
  `432163e9`; isolated upload-failure cleanup retained by `77a58de7`.
- [Fork PR #78](https://github.com/thomas9120/gufo-windows/pull/78) preserves
  ancestry. The [deferral record](#pr-688-windows-dflash2-continuation-latency)
  and machine-readable guard prevent automatic reintroduction. #699 stays reverted.

The final production difference from the Windows base is only the DFlash2
context-upload scope guard. Its unchanged upstream test delays a host upload,
injects a recoverable device-copy error, overwrites caller data immediately on
return, checks the uploaded bytes and verifies recovery, for pageable and pinned
sources. A Windows-only test-process HIP import hook replaces ELF linker wrapping;
it is absent from production binaries. Linux keeps the upstream fixture and hook.

All four full-import candidate profiles individually pass `long-context`,
text/thinking/tool/image cancellation, exact RAM replay and disk restart/spacing;
actual drafts are observed in DFlash2 and MTP. These correctness results do not
clear the failed matched DFlash2 timing/continuation comparisons. Flash-Next
AR/MTP match baseline output/work but their timing remains inconclusive.
The full-import CPU 17/17, numerical prefill, native cancel/resume, WDDM,
pinned buffer/transfers and Flash cross-slot probes pass. Those results refer
to the rejected optimization; final-subset checks are recorded below.

The extra broad component-cache fixture fails at its orphan assertion because
`std::ofstream` creates an unprotected DACL; existing Windows privacy checks
exclude that file from deletion. Original failure, ACL and source evidence are
retained. Upstream pinned-budget assertions passed independently, but their
temporary native adapter is removed with the optimization.

Final-subset validation, built clean at `852e2455` with TheRock 10.0.0/Clang
23.0.0, MSVC 14.51.36231 and SDK 10.0.26100.0:

- Native CPU build and **16/16 CTests pass**; HIP executable and affected test
  targets build. The temporary pinned-budget target is removed with the
  optimization, so its earlier 17-test run is not the final gate.
- Upload-lifetime fixture passes for pageable and pinned sources: no upload
  remains pending at failed return, original source bytes are preserved and
  subsequent injection recovers.
- Numerical prefill/captured-row checks and exact AR/DFlash2 native cancelled
  prefill resume pass. The latter retains the same eight-token frontier.
- Bounded Qwen DFlash2 HTTP RAM smoke passes at context 8192, one session and
  effective 1 GiB cache: cold 2,294/2,294 prompt/prefill tokens; retry reuses all
  2,294 with zero prefill. Both responses match and each proposes 18/accepts
  17 drafts. Private server shutdown is clean.
- Standard `gufo bench -p 2048 -n 128 -r 1 -v` AR/DFlash2 spots complete on
  baseline and final subset, at one common executable path. All four output
  hashes match (`095434b2...`); both DFlash2 arms report acceptance 0.439.
  These single-repetition spots do not qualify a speedup or full performance.
- Formatting checks 622 C++ files. Docs/dependencies, maintained-difference and
  sync inventory checks pass; all four #688 and six #699 excluded identifiers
  remain absent, and the recorded official head/reverts remain ancestors.

Production executable SHA-256:
`c8e66f755e4af759be88500ea253cc2a532f740c3989131a41c1b23d52861740`.
Later documentation/guard records do not change that production source.
Hosted qualification is pending on the final PR head. Evidence
([evidence archive](#evidence-archive)): `sync-688-2026-10-23/`; no full model sweep, speedup or full Windows
performance qualification is claimed.

## October 23, 2026: PR #709

Status: **local Windows checks and six executed hosted gates passed on
`96a67c74`; merged by PR #77 as `d812ebab`**. Three GPU/workflow jobs were skipped.

- Windows base: `4eb2aedd572be9b54125f4e49e8de04e216872a1` (PR #76 merged).
- Previous official ancestor: `2947624b9e65f52d4b4fdd96297f39e040a0ed24`.
- Imported official head: `3aa068b419e6272482298c1c6daccafb122764fa`.
- Integration merge: `be07d64b63ce5ecb9eab0c2d16d80a9e48eec668`.
- Scope: #709 only; the later upstream commits, including release changes,
  are outside this layer.
- Evidence ([evidence archive](#evidence-archive)): `sync-709-2026-10-23/`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#709](https://github.com/gufo-org/gufo/pull/709) | Native OS page-size and available-memory leaf | `3aa068b419e6272482298c1c6daccafb122764fa` |

### Merge review

One conflict in `HostAvailableBytes()`: adopt the upstream
`platform::AvailableMemoryBytes()` call while retaining the Windows early
return before Linux CMA/cgroup refinements. Upstream drops `<unistd.h>`, so
keeping the old fork helper call would also leave its declaration unavailable.
The native leaf uses the same `GlobalMemoryStatusEx` API and result as the
previous Windows helper. Host physical RAM remains separate from WDDM device
headroom, and host reserve calculations are unchanged.

The root CMake source-list addition merges cleanly and was reviewed. Both new
`os_info` files remain byte-identical to official. The existing Windows-owned
host test now checks the page size against `GetSystemInfo` and the new memory
query against surrounding native observations, alongside its existing
snapshot-budget and ceiling checks. No inference, n-gram or transfer code
changes. PR #699 remains reverted and its forbidden identifiers stay absent.

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Reviewed/unchanged: reopening/read flags and model paths are untouched. |
| WDDM admission budgets | Device helper reviewed/unchanged. Host page/memory queries and reserve calculations pass native OS-oracle checks; public MTP serving honors the requested 1 GiB RAM cache. |
| Pinned memory and transfer lifetime | Reviewed/unchanged; the bounded MTP RAM retry replays exactly with full prompt reuse. |
| Sockets and shutdown | Native HTTP/WebSocket/console checks pass in the CPU gate; the live HIP smoke server exits cleanly. |
| Filesystem identity, privacy and failures | Native CPU filesystem/store checks pass; legacy disk guards are retained. RAM-only startup does not populate the durable model-identity cache. |
| GPU queue policy | Reviewed/unchanged; CPU startup-policy checks pass and native HIP serving starts successfully. |

- Native Windows CPU configure/build and **16/16 CTests pass**.
- Native HIP `gufo` builds at `be07d64b63ce`; its SHA-256 is
  `b2df4482c6b1601f631b005841087af55115cdb255b8fb84cc2c8f9689901165`.
- Bounded MTP RAM startup/retry passes with actual drafting and the requested
  cache capacity; [model evidence](models/qwen3.8-flash-next/WINDOWS.md#os-info-adoption-october-23-2026).
- Formatting checks all 620 C++ files. Documentation, dependency inventory,
  maintained-difference and reverted-identifier checks pass; official ancestry
  is preserved.
- Linux/repository and hosted Windows checks remain pending until the sync
  PR completes. This platform refactor does not refresh numerical sweeps or
  inference benchmarks; the previous timing limits remain visible.

## October 23, 2026: PR #711

Status: **imported and locally qualified on `d87a8ae14a`; the hosted
Linux/repository and Windows CPU gates passed on `fb8f87a3`; draft
[PR #82](https://github.com/thomas9120/gufo-windows/pull/82) makes official
ancestry visible and waits for a merge-commit integration**. The layer is
additive: no production source changed, so no model, prefill or cache timing
work is claimed or required. The following record-only commits do not change
that production source.

- Windows base: `5606d1c67503f28e3a69620ca9be825561efc059` (PR #81 merged).
- Previous official ancestor: `8a79230617490ff374b38edc002a8e68d241cf68`.
- Imported official head: `7c8196527ad02ac4c585a32b1131fab7e70509d0`
  (upstream's squash merge of #711; its only parent is the previous ancestor).
- Integration merge: `d87a8ae14afd321cadac18b961356842061f83dc`.
- Scope: #711 only. The later upstream commits (for example #713 shutdown
  signals, #729 publish/prefault, #734 and #738 mapped files, #739 draft
  selection, #747 Nix caching) are outside this layer.
- Evidence ([evidence archive](#evidence-archive)): `sync-711-2026-10-23/`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#711](https://github.com/gufo-org/gufo/pull/711) | feat(platform): add the net leaf | `7c8196527ad02ac4c585a32b1131fab7e70509d0` |

### Merge review

Upstream adds the `src/core/platform/net.{hpp,cpp}` leaf and rewrites the
socket call sites in `http_server.cpp`, `websocket.cpp` and two CLI tests to
use it. This fork already has its own socket layer,
[`socket.hpp`](../src/core/platform/socket.hpp), so the merge decides which
layer owns the call sites: the fork's `gufo::platform` wrappers stay, and the
upstream leaf is imported byte-identical and left unreferenced for a later
slice.

- 45 conflict hunks across four files
  (`http_server.cpp` 19, `http_server_test.cpp` 18,
  `audio_websocket_test.cpp` 5, `websocket.cpp` 3), all of them the same
  mechanical choice. Each discarded hunk was reviewed individually; they
  contain only layer swaps:
  `MSG_NOSIGNAL` send to `net::SendNoSignal`, `::poll` to `net::PollSocket`,
  listener `fcntl(O_NONBLOCK)` to `net::SetNonBlocking`, the 120 s `timeval`
  pair to `net::SetSocketTimeouts`, `TCP_NODELAY` and the 5 s send timeout to
  `net::SetNoDelay`/`net::SetSendTimeout`, `::close` to `net::CloseSocket`,
  and the tests' `int fd`/`fd >= 0`/`net::` helper swaps.
- Reasons for keeping the fork's layer: upstream's send, receive and poll
  wrappers do not map WinSock errors to `errno`, so the existing
  `EINTR`/`EAGAIN` checks would stop working on Windows; it has no
  cancel-on-shutdown counterparts to `ShutdownSocket`, `StopSocketReads` and
  `CloseConnection`, which back the validated Ctrl+C/Ctrl+Break and orderly
  close contracts; and its call sites keep handles in `int`, which narrows a
  pointer-sized `SOCKET`. `net::Socket` itself is already width-correct, so
  these are call-site gaps rather than a reason to reject the leaf.
- `websocket.cpp` keeps its `<netinet/in.h>`, `<netinet/tcp.h>` and
  `<sys/socket.h>` includes: upstream deleted them because `net.hpp` supplies
  those declarations, but the retained fork wrappers need them and this
  translation unit does not include `net.hpp`. The Windows shims behind them
  are unchanged.
- `audio_websocket_test.cpp` and `http_server_test.cpp` had one pure-deletion
  hunk each (upstream's `namespace net = gufo::net;` alias). Both files are
  marker-enforced in [`shared-edits.json`](../tools/windows/shared-edits.json),
  and a pure deletion cannot carry a marker, so each alias became
  `// WINDOWS: keep native Winsock handles; gufo::platform wrappers replace
  net::.` Those two comment lines are the entire source delta of this sync
  against the Windows base.
- Clean merges reviewed: the root `CMakeLists.txt` source-list line adds
  `src/core/platform/net.cpp` to `gufo_core`; the fork's Windows overrides in
  `cmake/Windows.cmake` are untouched. `net.hpp` and `net.cpp` are
  byte-identical to official (blobs `81679b39`, `588a2b20`) and now compile on
  Windows in both presets, but no fork code references them yet.
- Follow-up: adopt the upstream leaf, or the parts of it that fit, once it
  covers native-width call-site handles, `errno` mapping and
  cancel-on-shutdown; see the
  [upstream friction TODO](todo.md#1-upstream-portable-fixes-to-shrink-the-48-file-src-delta-defer-this-for-now).

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Reviewed/unchanged: no engine, loader, n-gram or I/O path is edited; PR #699 stays reverted and its identifiers are absent. |
| WDDM admission budgets | Reviewed/unchanged: no device-memory or budget code; the live HIP server starts and stops normally. |
| Pinned memory and transfer lifetime | Reviewed/unchanged: no pinning or transfer code; PR #688 stays deferred and its identifiers are absent. |
| Sockets and shutdown | Validated: production paths keep the fork's layer. Native CPU gate passes `http_server_test`, `audio_websocket_test` and `win32_server_shutdown_test` (real Ctrl+C/Ctrl+Break during idle, active and partial requests), and the live HIP server exits cleanly on Ctrl+Break. |
| Filesystem identity, privacy and failures | Reviewed/unchanged: native filesystem and store checks pass unchanged. |
| GPU queue policy | Reviewed/unchanged: startup-policy checks pass and native HIP serving starts. |

- Native Windows CPU configure/build (176 targets) and **16/16 CTests pass** in
  18.3 s.
- Native HIP `gufo` builds at `d87a8ae14afd` (274 targets, TheRock 10.0.0); its
  SHA-256 is
  `64dfde137bb49232a519e13586f372fd9391ee141461df3e0005e7e59c50f891`.
- Live HIP smoke: ready in 26.703 s, `/health` ok, one buffered and one
  streamed completion (26 SSE events, terminal `data: [DONE]`), Ctrl+Break
  shutdown in 1.264 s with exit code 0.
- Formatting checks all 624 C++ files with clang-format 21.1.8. Documentation
  (101 files, 654 links), inventory, maintained-difference and
  reverted-identifier checks pass; `git diff --check` is clean and official
  ancestry is preserved.
- Hosted gates on `fb8f87a3` (draft PR #82): CPU contracts & repository
  checks (hosted Linux/Nix) 8m14s, Native CPU contracts 6m01s, Upstream
  difference guard 1m08s, Installer checks 46s, Browser launcher checks 51s and
  Conventional Commit title 5s. The hosted Linux build compiles the leaf's
  POSIX branches, which have no local build on this host. The advisory devlab
  functional checks were skipped because they need maintainer authorization;
  they do not gate this layer.
- No numerical sweep or inference benchmark is refreshed: this layer changes no
  model, prefill or cache behavior, so the previous timing limits stay as
  recorded. Integrate with a merge commit, not a squash or rebase, to keep
  official ancestry on `windows-main`.

## October 23, 2026: PR #350 qualification follow-up

Status: **no confirmed YaRN regression after matched controls; timing remains inconclusive**.
PR #75 merged as `a63f2b4e597a2ece22f817ec6164ff96149abdfe` while local
qualification was still running. The final October 22 record is
`sync-350-2026-10-22/FINAL-VALIDATION.md` in the
[evidence archive](#evidence-archive); follow-up commands,
hashes and per-request results are in `sync-350-2026-10-23/`.

The original two API pairs per mode used equal effective 8 GiB RAM budgets.
They preserve outputs and cache/prefill work, but fail two AR timing cells and
seven MTP cells. Later October 22 controls had unequal effective budgets as
host memory availability changed. One MTP pair consequently differs in warm
image-prefix reuse (13/5 candidate prefill tokens versus zero on baseline).
Those controls cannot clear the original matched failures.

On October 23, the retained executable hashes still match. Fresh baseline
controls verify the full 8 GiB at startup and restart: AR restore timing becomes
inconclusive, while three MTP total-duration cells remain outside the gate.
Decode time and actual token/draft work agree. An instrumented completion probe
and a byte-identical candidate under a different filename vary in latency;
these are diagnostic controls, not qualified fixes. A same-path baseline/head
ABBA control also reproduces the extra duration on the unchanged baseline.
Its standalone comparison flags one snapshot phase; the retained baseline
already varies from 12.30 to 30.30 ms in that exact history.

The final assessment retains all five valid matched MTP pairs, verifies their
build roles by SHA-256 across executable paths, and verifies all 143 runtime
files against the original control. It uses the unchanged upstream
`compare_runs` and `qualify` functions: **577 timing cells pass, 283 remain
inconclusive, none is confirmed slower; output and work changes are zero**.
Clamped and instrumented runs are excluded from qualification and retained
separately. Raw reports are unmodified. Original failed comparisons remain
failed observations; the assessment adds the verified controls instead of
discarding them or averaging requests. See the
[model measurements](models/qwen3.8-flash-next/WINDOWS.md#yarn-matched-follow-up-october-23-2026).

There is no confirmed #350 blocker requiring deferral, and no inference change
or relaxed tolerance is retained. A complete Windows performance pass remains
unclaimed. PR #699 remains reverted.

The Windows functional comparison wrapper now rejects clamped requests and
different effective cache budgets before invoking the upstream comparator.
Its native-host unit checks also cover restart clamping and staging/disk
capacity differences. It preserves the original per-request 5% and 3 ms gates
and retains rejected evidence.

## October 22, 2026: PR #350

Status: **merged; correctness evidence completed, Windows timing unqualified**.
See the October 23 follow-up above for the latest investigation status.

- Windows base: `d10a852f0f094d926128ecf2e9dd112cae0c4a8e` (PR #74 merged).
- Previous official ancestor: `53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe`.
- Imported official commit: `2947624b9e65f52d4b4fdd96297f39e040a0ed24`.
- Integration merge: `8170cfc855484da939df61cbcf4c31d125cde772`.
- Scope: #350 only. The later #709, #688, #710, #677 and #711 commits will
  follow in separately qualified sync PRs.
- Evidence ([evidence archive](#evidence-archive)): `sync-350-2026-10-22/`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#350](https://github.com/gufo-org/gufo/pull/350) | Opt-in Flash-Next YaRN context extension | `2947624b9e65f52d4b4fdd96297f39e040a0ed24` |

### Merge review

One conflict, in `projection_ops_test.cpp`: preserve the null vision descriptor
from the #699 revert and pass the new YaRN argument. Clean merges of the
executor, kernels, engine and serving options were also reviewed. #699 remains
reverted; its six forbidden identifiers stay absent.

Windows' MSVC library prints padded hexadecimal fractions in `std::hexfloat`,
breaking upstream's pinned YaRN cache identity. The Windows-only normalization
in `rope_scaling.cpp` removes that padding and passes the original test
unchanged. Linux formatting is unchanged. The shared edit is marked and listed.

The Windows-owned `win32_yarn_snapshot_test.cpp` covers checkpoint replay
without the upstream snapshot test's Linux ELF fault hooks. It checks exact
logits, tokens and actual MTP draft work, serialized replay, old executor-format
rejection/reset/recovery, and rejection of a different scaling factor.
Bitwise replay controls use matching prefill boundaries. The current executor
format changes from `QFNSNAP5` to `QFNSNAP6`; the outer session version stays 18.

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Windows reopening/read flags unchanged; native n-gram checks pass. Fresh-process inference timing remains inconclusive pending controls. |
| WDDM admission budgets | Helper unchanged; native 409,600-capacity AR/MTP fixtures and public 262,145-capacity MTP serving pass without weakening admission. |
| Pinned memory and transfer lifetime | Native snapshot-copy operator, AR/MTP 4K cross-slot checks and above-native YaRN checkpoint replay pass. |
| Sockets and shutdown | Implementation unchanged; bounded native HTTP/WebSocket and real console-shutdown checks pass. |
| Filesystem identity, privacy and failures | Native implementation and model payload/factor compatibility checks pass. Scaled MTP API disk restart passes exact response replay and actual drafting. |
| GPU queue policy | Boundary unchanged; bounded native startup policy checks pass. |

Completed local checks and current limits:

- Native Windows CPU gate: **16/16 pass** (`final-cpu-ctest.txt`).
- Selected operators: eight pass initially; the original rope-policy test fails
  on MSVC padding, then passes after the normalization (`head-operator-ctest.txt`,
  `fixed-rope-ctest.txt`). No GPU kernel change was needed after those passes.
- Full-model numerical/continuation evidence and per-request timing details are
  recorded in the [Windows model sidecar](models/qwen3.8-flash-next/WINDOWS.md).
- Final bounded YaRN fixture: AR/MTP exact replay and scaling-factor rejection
  pass (`final-yarn-snapshot-3.log`). Earlier fixture assumptions and their
  failed observations remain retained; the production restore/reset behavior
  is byte-identical to the Windows base.
- Shared formatting passes for 618 C++ files. Documentation, dependencies,
  maintained-difference and reverted-identifier checks pass.
- First-prompt control: correctness passes in all eight fresh processes;
  snapshot timing and first MTP decode have noisy cells. Overall timing is
  **inconclusive**, not passed (`cold-control-1/`). An unchanged-base MTP
  control also reports moved snapshot cells on identical binaries.
- Standard AR/MTP pp2048/tg128 spots, extended native YaRN replay and public
  scaled MTP startup/RAM retry/disk restart pass. See the model sidecar for
  measurements and scope. Both CLI opt-in refusal checks pass.
- All 14 selected API executions pass their own correctness assertions, but
  the original matched timing failures and later unequal-budget cache-work
  failure remain retained. They do not establish a performance pass.
- Six executed hosted Linux/repository/Windows CPU gates pass on `5d60b5bc`;
  skipped GPU/functional/report jobs are not counted as passes. No complete
  Linux GPU/model or Windows performance qualification is claimed.

## October 22, 2026: PR #702

Status: **candidate local checks pass; hosted qualification is recorded on the sync PR**.

- Windows base: `e416eac00b1450eb03d82165d76ebd1970ac0364` (includes PR #73).
- Previous official ancestor: `e060aeed923730b74ea3f4116c3c39de55741370`.
- Imported official commit: `53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe`.
- Integration: ordinary merge, `45bb338c2e5e42b5d99a985c65be22e05792c1fb`.
- Fetched `official/main`: `7c819652`. This import intentionally stops at
  #702 as requested; the six newer upstream commits are not included.
- Evidence ([evidence archive](#evidence-archive)): `sync-702-2026-10-22/`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#702](https://github.com/gufo-org/gufo/pull/702) | Stop the benchmark server when startup is interrupted | `53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe` |

### Merge review

No conflicts or files changed on both sides. The two imported files,
`tools/gufo/model_bench/servers.py` and `tests/tools/serving_bench_test.py`,
remain byte-identical to the recorded upstream commit. The helper now stops
its server and re-raises when readiness checking exits through an exception,
including `KeyboardInterrupt`. Review covered early process exit, readiness
timeout and the existing process-group shutdown. No engine, model, CMake,
dependency or Windows platform code changes.

The inventory finds no incoming change to #699's reverted regions. Its revert
and reintroduction requirements remain intact; the candidate check confirms
that its six forbidden identifiers remain absent.

### Compatibility and validation

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Reviewed/unchanged: no model or I/O path changes. |
| WDDM admission budgets | Reviewed/unchanged: no admission or device-memory changes. |
| Pinned memory and transfer lifetime | Reviewed/unchanged: no allocation, transfer or snapshot changes. |
| Sockets and shutdown | Native engine paths unchanged; bounded HTTP/WebSocket and real console-shutdown checks pass. Benchmark-helper interruption cleanup passes separately under WSL. |
| Filesystem identity, privacy and failures | Implementation unchanged; bounded identity, privacy and failure checks pass. |
| GPU queue policy | Implementation unchanged; bounded native startup policy checks pass. |

Completed checks:

- `tools/windows/build.ps1 -Preset windows-cpu -Vcpkg VCPKG_PATH
  -Ninja NINJA_PATH -Jobs 4`: pass with TheRock's compiler, MSVC 14.51.36231,
  SDK 10.0.26100.0, CMake 4.4.0 and the existing pinned vcpkg manifest.
- `ctest --preset windows-cpu --no-tests=error --output-on-failure`:
  **16/16 pass**, 19.49 seconds (`cpu-build.txt`, `cpu-ctest.txt`).
- `python3 -I tests/tools/serving_bench_test.py` under WSL/Python 3.12.3:
  **pass**, including the new real-process interruption test
  (`serving-bench-wsl.txt`). This helper uses POSIX process groups; this is
  Linux lifecycle evidence, not a claim of native Windows benchmark support.
- Matched interrupted-start control with the exact Windows-base helper and
  candidate helper under WSL: both propagate `KeyboardInterrupt`; the base
  leaves its real `sleep` child alive, while the candidate has already reaped
  it. The control explicitly cleans up both test-owned processes
  (`startup-control.py`, `base-servers.py`, `startup-control.txt`).
- Inventory before and after merging, including `-Candidate HEAD`: pass.
  `diff-guard.ps1`: pass, with 72 changed upstream files, unchanged by this
  import (`inventory.md`, `candidate-inventory.md`).
- `check-docs.py`, `check-dependencies.py`, `git diff --check`, and the
  imported-commit ancestry check: pass. No C++ edits require formatting.
- Hosted Linux/Nix and Windows CPU gates run on the draft sync PR; their
  results qualify that PR's exact head.

### Live checks and limits

No inference or shared runtime paths changed, so no GPU/model rebuild,
numerical sweep or inference timing comparison is required for this import.
The existing Windows qualification limits remain unchanged. The benchmark
cleanup result is supported by a real child process, without loading a model.
Preserve official ancestry by merging the sync PR with a merge commit.

## October 21, 2026

Status: **local handoff complete; #699 reverted; qualification limited below**
on `t3/sync-upstream-oct-21-prs`. The candidate is retained locally for review.
The maintained `windows-main` remains at the recorded base; it has not been
promoted past the outstanding qualification limits.

- Windows base: `db4f6fbcea0514db1456abf8571f20cb525de30e` (also
  `origin/windows-main` at fetch time; no unpublished divergence).
- Previous official ancestor: `f020bd3a`.
- Imported official head: `e060aeed923730b74ea3f4116c3c39de55741370`.
- Integration: ordinary merge of the recorded official head,
  merge commit `54fb4b4e`, followed by the explicit #699 revert `161035d8`.

| PR | Upstream change | Merge SHA |
| --- | --- | --- |
| [#692](https://github.com/gufo-org/gufo/pull/692) | Compare the bound publication waiter with cold by answer | `0e4f987b8145bc37fca25b601881abd369982c36` |
| [#540](https://github.com/gufo-org/gufo/pull/540) | Qwen 27B AR and DFlash2 cache continuation | `da5d5a2b237c654bc0523e33fdee10465240da0c` |
| [#690](https://github.com/gufo-org/gufo/pull/690) | Wide Flash-Next routes for prefill passes above eight rows | `53f73b5653b32caf1d8cb3113c3f2f05f0e88d0a` |
| [#699](https://github.com/gufo-org/gufo/pull/699) | Reduce Flash-Next sampled decoding overhead; **reverted/deferred on Windows** | `e060aeed923730b74ea3f4116c3c39de55741370` |

The only additional incoming commit is upstream's v0.20.0 release
`0c03748a`. Its version, manifest and changelog changes are retained as published.
No version is manually assigned by this sync.

### Merge review

Only `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` and `QUALITY.md`
conflicted: both branches inserted independent evidence at the top. Retained
the October 21 upstream sections and the October 20 Windows n-gram evidence.
No source file required conflict resolution. The later #699 revert conflicts
only in `QUALITY.md`; its resolution removes the reverted claims and preserves
the Windows October 20 n-gram evidence.

Reviewed the clean merges in `inference_backend.cpp`,
`text_model_runner.cpp`, `text_generation_scheduler.cpp` and the Flash-Next
ROCm executor: WDDM-aware resource claims, native host-memory accounting,
live prefill metrics and snapshot admission remain present. Shared snapshot
copy kernels moved to `src/core/hip/snapshot_copy.*`; their consumers and
both model build lists follow the move. Windows disk-store guards remain.

The shared transfer pool now supports a bound execution stream. Native pinned
transfer coverage additionally checks real copies on that stream, invalid and
repeated binding, and stream-owner retention after pool/lease destruction until
the completion releases it. Linux fault-injection harnesses remain unchanged.
The initial import's native run also exposed a stale pre-sync fixture: committed backing
now registers anonymous pages, but the test still queried `hipHostMalloc`
flags. It now checks the host registration and device mapping; staging still
checks `hipHostMallocCoherent`. The allocator is unchanged by this sync.
See [AMD's memory API contract](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html).

### Compatibility and validation

Validation results and pending qualifications are recorded here before handoff.
Its `build/sync-2026-10-21/` evidence was not archived and is no longer retained.
The previous Windows executable and colocated DLLs remain in the maintained
checkout; the candidate builds into this task's own `build/` directories.

| Contract | Outcome |
| --- | --- |
| Cached concurrent n-gram reads | Native independent row/hash/offset/descriptor checks pass. Windows still uses `O_CONCURRENT_RANDOM`; fresh-process AR/MTP controls below. |
| WDDM admission budgets | Validated: native live DXGI/HIP oracle and allocation/fallback checks pass. Existing admission helpers survive the merge. Multi-GPU switching remains unqualified. |
| Pinned memory and transfer lifetime | Validated: corrected/expanded native test passes, including bound-stream copies/ownership. Flash-Next AR/MTP 4K cross-slot probes reproduce every captured component, logits, next-step tokens and actual MTP drafts. Fault injection and physical memory pressure remain unqualified. |
| Sockets and shutdown | Implementation unchanged; all bounded CPU transport and real console-shutdown checks pass. |
| Filesystem identity, privacy and failures | Implementation and both checkpoint-store guards unchanged; bounded CPU identity/privacy/failure checks pass. Disk persistence remains deferred. |
| GPU queue policy | Windows boundary unchanged; CPU and HIP startup checks pass. No Windows queue-residency/performance claim. |

Completed checks (TheRock 10.0.0/Clang 23.0.0, MSVC 14.51.36231,
SDK 10.0.26100.0, CMake 4.4.0; compiler/cache details in the retained logs):

- `tools/windows/build.ps1 -Preset windows-cpu -Vcpkg VCPKG_PATH` and
  `ctest --preset windows-cpu --no-tests=error`: **15/15 pass** on the import (17.83 seconds) and again after the revert
  (16.91 seconds).
- `tools/windows/build.ps1 -Preset windows-gpu -Vcpkg VCPKG_PATH`: production
  HIP executable builds. Both builds use this checkout's pinned vcpkg manifest.
- Affected extra CPU checks: sampling, idle spill, Qwen continuation layout,
  Flash-Next n-gram, MTP sampling and prompt lookup: **6/6 pass**.
- Focused GPU gate: WDDM, pinned transfers, queue policy, speculative verifier,
  Qwen attention policy/long-context operators, Flash-Next projections,
  routed experts, snapshot copies and sampled-verification oracle: **10/10
  pass across the initial run and the focused pinned-fixture rerun**. The
  original failed pinned-fixture result is retained, not overwritten. After
  reverting #699, the remaining focused gate passes **9/9** (71.69 seconds);
  #699's removed sampled-verification test is not counted as a current pass.
- Qwen 27B continuation probe at 32K: **pass**, including component bytes,
  restoration and next-step logits/tokens. Native `--prefill-cancel-only` with
  the official DFlash2 sidecar: **AR and DFlash2 pass** exact resume against
  direct cold execution, with actual drafting. The broader
  `--continuation-only` includes Windows-disabled disk persistence and remains
  outside this qualification.
- Flash-Next continuation probe: `--mode ar` and `--mode mtp`, each with
  `--context 4096 --sessions 2 --backing-gib 1 --round-trip`: **pass**.
- Post-revert Flash-Next `qwen38_flash_next_session_test` with
  `--sampling-only` and `--cache-only`: **both pass**. Coverage includes graph
  capture/execution, EOS and sampling variants, 200-token checkpoint replay,
  and learned-policy greedy/thinking/nonthinking continuation with exact
  tokens, sampler state, drafts and final logits after RAM/serialized restores.
- Standard post-revert `gufo bench -p 2048 -n 128 -r 1 -v` with
  `--speculative off`, then `--speculative mtp --mtp-model MTP_PATH`:
  **both complete with exit zero**. AR pp/tg: **1,807.04/28.88 tok/s**;
  MTP pp/tg: **1,753.56/38.65 tok/s**. Greedy tg128 hashes match:
  `5177a75da6988d5ce9a14e1a744ed16829784eda18d40139d8a1e76469af8412`.
  MTP proposes 109 and accepts 58 drafts. These single-repetition spots do
  not qualify a branch speedup. Commands/results are retained in
  `final-bench-*-command.json`, logs and `final-bench-summary.json`.
  Windows PowerShell wraps verbose native stderr in `NativeCommandError`
  records; the captured benchmark exits are zero.
- Shared `tools/ci/check-format.py` with clang-format 21.1.8,
  `tools/ci/check-docs.py`, `tools/ci/check-dependencies.py`, and
  `git diff --check`: **pass**.
- Functional-runner unit tests under WSL: **80/80 pass**; new Qwen recovery
  evidence-refusal tests: **6/6 pass** on Windows and WSL. The initial Windows
  runner invocation hit a default-encoding failure (the affected case passes
  with Python UTF-8 enabled) and the existing POSIX-only `SIGHUP` harness.

The additional independent `qwen38_flash_next_gpu_probe --mtp-audit` fails
on image sequence 1, step 0: head-mixer relative RMS `0.0000695` exceeds the
unchanged `0.00005` scalar-oracle limit (maximum absolute error `0.0047131`).
The exact pre-sync Windows base, rebuilt with the same toolchain and the
byte-identical audit fixture, reproduces every printed value and failure.
This is an inherited Windows numerical qualification gap, not a new sync
failure. The oracle is not passed, its tolerance is unchanged, and both
`final-mtp-audit.log` and `baseline-mtp-audit.log` are retained.

Additional Windows harness exclusions, retained as failures in the evidence:
`cache_slot_test` requires POSIX `fork`/`SIGABRT`; `component_text_cache_test`
and `text_model_runner_test` reach the deliberately disabled disk stores.
These broader tests are outside the bounded Windows preset. They do not pass
on Windows and were not made to pass by weakening the store guards.

### Live checks and limits

The initial import's Flash-Next AR and MTP pass the SDK `long-context` and `cache-depth`
suites with 32K contexts, two sessions and an explicit 8 GiB RAM budget.
They exercise multi-turn recall, exact warm/cold answers, deep RAM reuse,
rewritten branches and peer rotation. MTP responses contain actual proposed
and accepted drafts. Each suite retains all request/phase timings; these
correctness runs do not by themselves qualify speed.

After the #699 revert, Qwen 27B AR and official DFlash2 also pass both API
suites at 32K/two sessions/8 GiB. DFlash2 responses contain actual drafts
(up to 115 proposed tokens in the long-context observations); both private
test servers exit cleanly with code zero. These runs exercise the #540
component-cache path rather than Windows-disabled disk persistence.

The additional `sampling-defaults` suite passes its explicit/default sampler
output checks but fails its zero-prefill replay assertion after the deep-cache
history: 15 of 22 prompt tokens are reused, and seven are re-prefilled.
Both exact pre-sync AR and MTP baselines reproduce the same output and
seven-token failure under the same deep history. Fresh MTP baseline and
import processes pass this suite. This is an inherited history-sensitive
cache assertion, not evidence of a new sampling regression; no blanket
sampling/cache or timing pass is claimed. The baseline MTP long-context
control also flags a reasoning-signature replay despite a correct answer;
the import passes that check. All raw observations remain retained.

The varied fresh-process prompt contains 11,036 tokens. Each process sends it
once with default prompt reuse and then with `cache_prompt: false`; both
requests report zero reused tokens. The initial AR prefill times are
6,533.72/6,093.59 ms (base) and 6,298.96/6,072.70 ms (initial import), with identical
completions and prompt/cache/draft counts. The repeated AR baseline also passes
the strict request-phase limits against that import. Its cumulative idle-copy
counter covers only 25.69 MiB at the second response, versus 72.48 MiB on the
import; the initial baseline covers 30.67 MiB. Those asynchronous background
totals are different work, remain visibly inconclusive in the raw comparison,
and do not qualify background-copy performance.

Controls use the exact Windows base built separately in the retained detached
`baseline-source` worktree, TheRock 10.0.0 for both revisions, byte-identical
runtime DLLs and one common `run-bin/gufo.exe` path. Builds finish before these
HTTP timing runs. Binary SHA-256 values, commands, loaded `/props`, complete
responses, phase counters, individual timings and graceful shutdown results
are in `identities.json` and the per-arm JSON/log files. Model identity uses
path/size/mtime; full model hashes were not collected. The initial import reports `db4f6fbcea05-dirty`; the final tested code is
commit `161035d8` (with documentation edits present during the build). Its
SHA-256 is `447196666e8f2e1158e78ac59ab2b464ee58469d2a84226ed1dec619e1419298`.
Both use upstream's 0.20.0 version; retained patches and binary hashes
distinguish every arm.

The rebuilt revert passes the affected MTP first-decode gate. Final AR
prefill measures 6,309.95/6,109.18 ms and decode 42.65/35.24 ms; these phases
also pass against the focused base control, with identical output and work.
Snapshot timing remains unqualified: final MTP first snapshot is 67.92 ms
versus 61.66 ms, and final AR snapshots are 53.51/40.56 ms versus 42.07/35.57 ms.
Those flags and asynchronous copy totals remain in
`reverted699-comparison.json` and `final-ar-comparison.json`; no complete
performance pass is claimed from these pairs.

Full Windows inference/performance qualification remains pending. Linux/Nix
checks could not run: WSL has Python but no Nix/CMake/compiler toolchain, and
no hosted check was triggered for this local-only handoff. Qwen Q8_K_XL and
the DSpark/DeepSeek profiles are unavailable locally and remain unqualified.
Multi-GPU, physical memory pressure and Windows disk persistence also remain
outside the evidence above. The upstream PRs' Linux measurements do not
qualify Windows. PR #699 is explicitly reverted for the Windows regression;
its prior Linux evidence does not override this disposition.
