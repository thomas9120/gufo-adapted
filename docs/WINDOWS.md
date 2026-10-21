# Windows (native, gfx1151)

Gufo builds and runs natively on Windows 11 x64 against AMD's TheRock ROCm
distribution for gfx1151 (Ryzen AI Max+ 395 / Radeon 8060S). No WSL. Platform
adaptations live in `compat/win32`, Windows-only files and `_WIN32` branches.
The fork also changes shared engine code; Linux CI remains necessary.

Follow the [upstream sync workflow](UPSTREAM_SYNC.md) to import official fixes
while preserving Windows behavior. Prioritize fixes over feature updates.

## Guided setup

After cloning, double-click [setup-windows.bat](../setup-windows.bat) in the
repository folder. It shows its plan before installing anything. Enter `y` to
continue, and accept the relevant installer prompts. Setup builds Gufo and opens
the [web launcher](../tools/gui/README.md). Later, use
[launch-gui.bat](../launch-gui.bat). Models are selected separately in the GUI;
setup does not download them or start inference.

Setup requires native Windows 11 x64 on Ryzen AI Max/Strix Halo. Install the AMD
graphics driver first; GPU memory allocation remains a manual AMD Software
setting. Plan for at least 25 GiB free on the installation/build drives for fresh
setup, plus space for models. An existing-tool rebuild requires 5 GiB free.

- Compatible installed tools are reused. New WinGet installs select Git 2.55.0.3,
  VS Build Tools 18.8.0 with the C++ tools and Windows SDK, CMake 4.4.0,
  Ninja 1.13.2, and 64-bit Python 3.14.6. Package licenses and Windows elevation
  prompts remain visible. If WinGet is unavailable, install/update Microsoft's
  [App Installer](https://aka.ms/getwinget), or install the prerequisites below.
- The compiler/runtime download is TheRock **10.0.0**, verified using the SHA-256
  below. A new vcpkg checkout uses the commit pinned in `vcpkg.json`.
  Downloads and managed dependencies live under `%LOCALAPPDATA%\Gufo\dependencies`.
  Existing TheRock/vcpkg installations are reused without updating their checkouts.
- Detected tool paths are saved in ignored `build\windows-setup.json`. Interrupted
  downloads, completed installs and incremental builds are reused on rerun.
  Setup logs are `build\setup-YYYYMMDD-HHMMSS.log`; a failure keeps the batch
  window open. Fix the reported issue and rerun, including after a requested reboot.
- Stop a Gufo instance using `build\release\gufo.exe` before rebuilding it.
  Setup refuses to overwrite that running executable. It checks `--version` and
  GPU diagnostics outside the build environment before preparing the GUI.
- Existing `%LOCALAPPDATA%\Gufo\launcher.json` settings are preserved, including
  a custom executable selection. New GUI settings default to this checkout's
  `build\release\gufo.exe`; opening the GUI still does not load a model.

Optional commands from the repository root:

```powershell
.\setup-windows.bat -CheckOnly       # preview, no writes/downloads/installs
.\setup-windows.bat -NoLaunch        # complete setup without opening the GUI
.\setup-windows.bat -Rocm D:\TheRock\build -Vcpkg D:\vcpkg
```

`-Yes` accepts the setup plan (installer/license prompts may still appear).
`-Jobs N` changes build parallelism; the default is 4. Rerunning setup builds
the currently checked-out source; it does not pull, reset or switch Git branches.
The manual scripts below remain available for developers.

## Updating and rebuilding

Use **Exit** in the GUI to close Gufo and the launcher, then double-click
[update-windows.bat](../update-windows.bat). It fetches `windows-main` from `origin`,
switches existing `windows-port` installs to local `windows-main`, and updates
that branch using a fast-forward-only merge. It then runs
`tools\windows\build.ps1` to rebuild
`build\release\gufo.exe` and stage its runtime DLLs and kernel libraries. For the
documented clone, future updates track `origin/windows-main` on GitHub.
The old local `windows-port` branch is retained.
It checks the rebuilt executable with `--version` before reporting success.
Open [launch-gui.bat](../launch-gui.bat) afterward to start the app.

The updater reuses the tools detected by setup, including paths saved in
`build\windows-setup.json`. Run setup first if prerequisites are missing.
Optional overrides work from a terminal:

```powershell
.\update-windows.bat -Jobs 4
.\update-windows.bat -Rocm D:\TheRock\build -Vcpkg D:\vcpkg -Ninja D:\tools\ninja.exe
```

Local changes and untracked files must be committed, stashed or moved before
updating; ignored build files and local settings can stay in place. A detached
HEAD, a branch other than `windows-port` or `windows-main`, Git failure or
diverged `windows-main` history stops the update before rebuilding. Unpublished
`windows-port` commits also block the switch. The script does not reset local
commits. The two remote branches do not need shared history for this transition.
If compilation fails after a successful pull, the updated source stays in place;
fix the reported issue and rerun. The batch window stays open on errors.
Existing GUI settings and executable selections are preserved; select this
checkout's `build\release\gufo.exe` in the GUI before restarting Gufo.

## Prerequisites

| Piece | Default location | Notes |
| --- | --- | --- |
| TheRock ROCm 10.0.0 (Windows, gfx1151) | `C:\TheRock\build` | [`therock-dist-windows-gfx1151-10.0.0.tar.gz`](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz), SHA-256 `1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`; extract so that `C:\TheRock\build\bin` exists. Its clang compiles C, C++ and HIP |
| vcpkg | `C:\vcpkg` | Run `bootstrap-vcpkg.bat`; CMake installs the dependencies pinned by `vcpkg.json`. Without `-Vcpkg`, `build.ps1` and `check.ps1` use the checkout setup recorded, then `VCPKG_ROOT`, then `C:\vcpkg` |
| Visual Studio Build Tools | any | "Desktop development with C++": MSVC STL + Windows SDK only; tested with VS 18 (MSVC 14.51) |
| CMake 3.21+ and Ninja | `PATH`, or Ninja at `C:\tools\ninja` | |
| Dedicated GPU memory | AMD Software > Performance > Tuning > Variable Graphics Memory | 96 GB for Qwen3.8-Flash-Next at 256K context; see [Memory](#memory) |

The port is validated on TheRock 10.0.0. Other releases usually build, but a
newer clang can round fused kernels differently (see [Test status](#test-status)).

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # what is missing, and how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # release -> build\release\gufo.exe
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test   # + tests and tools
```

`build.ps1` imports the MSVC environment, configures with TheRock clang and the
vcpkg toolchain, builds, and copies the ROCm and vcpkg runtime DLLs plus the
hipBLASLt/rocBLAS kernel libraries next to `gufo.exe`, so `build\release` runs
outside the developer shell. Runtime files with matching size and modification
time are reused on rebuild. Local MSVC runtime DLLs are staged as well. If CMake
selects Visual Studio's LLVM OpenMP library, its matching `libomp140.x86_64.dll`
is copied for local source-build use. **Do not redistribute this build folder**:
that OpenMP DLL comes from `debug_nonredist`; a public binary release needs a
distributable runtime and a separate dependency/license audit.

The manifest records the vcpkg baseline used for the Windows dependencies.
`build.ps1` installs them under `build\vcpkg_installed`, independently of other
projects' packages, and reconfigures on each invocation. The first build needs
network access and can take several minutes to build the dependencies.
Existing classic-mode CMake caches are cleared once during migration so they
cannot retain paths to the old global dependencies.
Git checkouts also embed the source revision in `gufo --version`, including a
`-dirty` suffix for modified tracked files. Record this output with benchmark
results. Source archives without Git metadata keep the `development` fallback.

## Running Qwen3.8-Flash-Next

Download the qualified files with the Hugging Face CLI
(`pip install -U huggingface_hub`):

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"
powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1
```

`run-flash-next.ps1` finds the files in the Hugging Face cache and serves an
OpenAI-compatible API on `http://127.0.0.1:8080/v1` (thinking on, adaptive MTP,
the model's own sampler). `-Think off`, `-Draft mtp3|off`, `-Context N` and
`-Mode bench` (pp2048/tg128 at depths 0..128K) are the common variations. The
server is ready when the log shows `event=load_completed`.

The script uses the pinned revision above and takes the target and MTP sidecar
from the same snapshot. Use `-Snapshot PATH` to select another model directory,
or `-MtpModel PATH` to explicitly select a separate sidecar. The printed command
records the selected paths. `GUFO_*` environment
variables (`GUFO_PLATFORM_TUNING`, diagnostics) reach the server; the script
lists any set in the calling shell.

## Qwen27B weight memory

Qwen3.8-27B uploads GGUF shards into `hipMalloc` device allocations on
Windows. The existing `hip::WeightUpload` pipeline reads the retained file
descriptors through bounded staging buffers (16 workers, 16 MiB each),
without creating a second full-model copy in host RAM. Tensor encodings,
offsets and inference arithmetic are unchanged. Linux retains its registered
host-memory path. DFlash2 already uploads its private weights to device
memory and shares the target's embedding/output pointers.

This is a correctness workaround, not an optional tuning switch. On
September 28, 2026, with TheRock 10.0.0 and Radeon 8060S/gfx1151, registering
a 17,559,178,144-byte host buffer (the UD-Q4_K_XL file size) returned success,
but the next `hipStreamCreateWithFlags` stalled. Creating the stream first
instead produced `hipErrorLaunchFailure` (719) when synchronizing a GPU read.
A standalone probe reproduced both failures without loading a model;
64 MiB registration and a device upload of the same large buffer passed.
The precise driver defect or size limit remains unidentified.

Low dedicated-VRAM use in this failure was not CPU-only inference: HIP was
trying to access registered host memory. The server never reached readiness.
Do not infer support from registration success or a small-allocation probe;
test full-sized weights, GPU access and model output. Use a matching BF16
projector for vision; F16 is not interchangeable.

The [Qwen27B quality record](models/qwen3.8-27b/QUALITY.md#windows-device-upload)
records the upload control, exact logits and AR/DFlash2 text/vision checks.

## How the port works

- `compat/win32/include` shadows the POSIX headers Gufo uses (`unistd.h`,
  `sys/mman.h`, `fcntl.h` additions, sockets, `spawn.h`, ...) and
  `compat/win32/posix.cpp` implements them on Win32. The directory is on the
  include path only on Windows. `_CRT_DECLARE_NONSTDC_NAMES=0` keeps the
  UCRT's own POSIX aliases out of the way.
- Semantics that matter:
  - `O_DIRECT` is `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, so the
    weight uploader keeps its parallel aligned direct reads.
  - NTFS serializes direct reads of a file that is also mapped, and Gufo maps
    every model shard: 32 threads of 4 KiB direct reads fall from ~70K to ~5K
    reads/s. Large sequential reads barely notice, but the Flash-Next n-gram
    table reader does ~16 random row reads per prompt token. On Windows it
    opens the table with `O_CONCURRENT_RANDOM` (a cached overlapped handle,
    68K reads/s at 32 readers, 128K at 128) and runs 128 readers. That took
    real-text prefill from ~300 to ~680-840 tokens/s.
  - `open("/proc/self/fd/N")` is `ReOpenFile`: an independent handle to the
    same file, immune to path replacement, exactly as on Linux.
  - `flock` locks one byte far past EOF (Windows locks are mandatory) and
    `close` unlocks explicitly (Windows releases locks asynchronously).
  - `renameat` uses POSIX-semantics rename; `st_ctime` is NTFS ChangeTime.
  - `posix_spawnp` passes descriptors through the CRT `lpReserved2` block, so
    FFmpeg's `pipe:N` inputs work.
  - Uncaught C++ exceptions and crashes print a Linux-style message (with
    `what()`) or `module+offset` for `llvm-symbolizer`.
- `src/core/platform/socket.hpp` covers the socket differences call sites
  cannot hide: `closesocket`, Winsock errors to `errno`, millisecond
  `SO_RCVTIMEO`, lingering close (Windows discards data queued before a reset),
  and polled websocket reads (a Winsock `SD_RECEIVE` resets the connection).
- `compat/win32/hip_include/cmath` works around MSVC 14.5x declaring
  `isgreater` & co. `constexpr` (hence `__host__ __device__` under HIP), which
  collides with clang's HIP math declarations.
- Executables reserve an 8 MiB stack, Linux's default (Windows: 1 MiB).

## Windows-validated switches

Some behaviour changes were developed and measured on Windows only. They live
in `src/core/platform/tuning.hpp`: each is on by default on Windows and off
elsewhere; these defaults do not qualify every shared change on Linux. The decode switches
change timing and memory placement only. `GUFO_PLATFORM_TUNING` overrides the
defaults on any platform, for
example `GUFO_PLATFORM_TUNING=+prompt_checkpoint` to try one on Linux, or
`GUFO_PLATFORM_TUNING=none` to run the Linux path on Windows.

| Switch | What it does | Measured on Windows |
| --- | --- | --- |
| `prompt_checkpoint` | Qwen serving keeps a checkpoint before the generation suffix for every prompt, not only with tools or without preserved thinking | A client that re-sends the previous turn without its reasoning hits the cache instead of re-prefilling everything: follow-up TTFT 0.52 s at 82K context |
| `copy_kernels` | Qwen3.8-Flash-Next decode copies (uploads, downloads, rollback, hidden carry) as small kernels instead of `hipMemcpyAsync` | A copy-engine hand-off costs ~30 us per graph node and ~150 us per compute/copy switch; rollback 8.5 -> ~2 ms, probe +7-8% with `recorded_rollback` |
| `recorded_rollback` | The speculative state rollback replays as one recorded graph per kept length | (with `copy_kernels`, above) |
| `keep_rollback_rows` | Rollback rows survive session resets and snapshot restores | Verify and rollback graphs are captured once per session instead of on every request |
| `flush_before_wait` | `hipStreamQuery` before blocking on n-gram rows: HIP on Windows only submits queued launches at a flush | Draft-to-verify GPU idle 1.9 -> 1.15 ms per cycle |
| `flag_waits` | Decode-sized passes wait on a completion flag in coherent pinned memory instead of `hipStreamSynchronize` | `hipStreamSynchronize` returns ~0.4 ms late after a large graph; ~0.6 ms per cycle with `verify_graph_candidates` |
| `fast_sampling` | Sampled decode reads GPU-selected top-64 candidate lists (`SamplerState::DistributionFromTop`) instead of full vocabulary rows, whenever the list provably holds the whole top-k | 3-6% less time per decode cycle |
| `verify_graph_candidates` | With `fast_sampling`, the verify graph selects the candidate lists itself | (with `flag_waits`, above) |
| `fused_hc_down` | The HC mixer down projection fuses its SiLU scale into the GEMV write and prefetches deeper for 2-8 tokens | One launch fewer per mixer; part of the 3-6% above |
| `hot_first_upload` | Weights read on every token are uploaded before the routed experts | GEMVs on memory allocated late run 6-22% slower on Windows; placement only |

The port author's Windows checks compared all decode switches on against all
off: the original `gufo bench --logit-eval` dumps matched at 4418 positions,
and one fixed-seed sampled decode produced byte-identical text. Those dumps
contain target/top-64 log probabilities and a normalization value, not every
raw logit. This does not establish full-row equality or rejection/rollback
correctness. In those measurements, all off decodes
7.6-9.5% slower on Windows (sampled probe, prose / code / reasoning 32.0 /
34.3 / 44.1 against 34.5 / 37.2 / 48.3 tok/s). `prompt_checkpoint` is the
exception to identical text: splitting prefill at the generation suffix
changes rounding the way a different prefill chunk size does, so sampled texts
differ from an unsplit prefill, as equally valid samples.

`--logit-eval` now also writes `PREFIX-sN.bin.sha256`, hashing every raw float
in each recorded vocabulary row. Compare two runs or schedules with:

```powershell
python tools/bench/logit-eval.py left-s0.bin.sha256 right-s0.bin.sha256
```

The comparison rejects incomplete files and mismatched targets; exit 0 means
all recorded rows match, 1 means differing rows, and 2 means invalid inputs.
These teacher-forced schedules fully accept each batch. They do not cover
rejection prefixes, snapshot interleaving, or output quality. Keep those checks
separate. Changing speculative policies can also consume random draws
differently, so distribution preservation does not promise identical text for
the same seed. IQ4 targets that re-quantize Q6_K tensors at load remain
unqualified and are not covered by lossless quantization claims.

## Memory

On Windows, HIP allocations are not limited to the dedicated carve-out that
the BIOS or AMD Software reserves; `hipMalloc` succeeds up to ~110 GiB on a
128 GB machine, with the rest backed by shared system memory. Measured on a
128 GB Strix Halo:

- `hipMemGetInfo` double-counts memory beyond the carve-out (free drops 2 GiB
  per GiB allocated), which made Gufo refuse to size its sessions.
  `src/core/platform/device_memory.cpp` reports the process's WDDM budget
  (DXGI `QueryVideoMemoryInfo`, adapter matched by the HIP device LUID)
  instead.
- Backing new GPU memory costs ~28 GiB/s inside the carve-out but only
  ~0.63 GiB/s beyond it (Windows commits and maps system RAM page by page),
  which also halves concurrent disk reads during the load.
- Weights beyond the carve-out are also slower to read: the decode
  matrix-vector kernels run 6-22% slower on them.

Qwen3.8-Flash-Next at 256K context needs ~91 GiB of device memory (weights,
KV cache, recurrent state, scratch for 4096-token prefill chunks). With a
96 GB carve-out all of it is dedicated: the load takes ~30 s and decode runs
at full speed. A 64 GB carve-out works, but the load takes ~2 minutes and
decode is slower.

## Test status

- The original port author reported all Flash-Next operator tests passing,
  plus 86/92 CPU-labelled tests. These are not results for every later commit.
- CI on `main` and `windows-port` includes a bounded Windows CPU suite and
  launcher checks, alongside the Linux checks. Hosted Windows CI does not
  validate GPU kernels or real-model quality.
- Not ported: Linux-only Python dev tools (`h3_profile*`, `qwen27b.tools`,
  TTS reference verification needs numpy), and `video_jobs_test` needs
  symlink privileges (Developer Mode).
- `qwen_ssm_ops_test` and `minimax_h3_dit_analytic` assert bit-identical
  fused vs. reference kernels; TheRock's newer clang rounds differently.
