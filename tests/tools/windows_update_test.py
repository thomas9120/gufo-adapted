"""Updater checks with local Git repositories and a stub build; no GPU or installs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == "nt", "Windows updater requires Windows")
class WindowsUpdateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binaries = tempfile.TemporaryDirectory(prefix="gufo update binaries ")
        cls.addClassCleanup(cls.binaries.cleanup)
        cls.engine = Path(cls.binaries.name) / "gufo.exe"
        source = cls.engine.with_suffix(".cs")
        source.write_text('''using System;
class Stub {
    static int Main(string[] args) {
        Console.WriteLine("gufo test revision");
        return int.Parse(Environment.GetEnvironmentVariable("UPDATE_VERSION_EXIT"));
    }
}
''')
        compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        subprocess.run([str(compiler), "/nologo", f"/out:{cls.engine}", str(source)],
                       check=True, capture_output=True, text=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo update & spaces ")
        self.addCleanup(self.temporary.cleanup)
        # Native process paths expand Windows 8.3 aliases used by CI's TEMP.
        self.root = Path(self.temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.shell = shutil.which("powershell.exe")
        self.git_exe = shutil.which("git.exe")
        self.assertTrue(self.shell and self.git_exe)
        self.environment = {key: value for key, value in os.environ.items()
                            if key.lower() != "psmodulepath"}
        self.environment.update(UPDATE_ENGINE_STUB=str(self.engine),
                                UPDATE_BUILD_EXIT="0", UPDATE_VERSION_EXIT="0")
        scripts = self.checkout / "tools/windows"
        scripts.mkdir(parents=True)
        for name in ("update.ps1", "setup.ps1"):
            shutil.copy2(ROOT / "tools/windows" / name, scripts / name)
        (scripts / "build.ps1").write_text(r'''param($Preset, $Target, $Jobs, $Rocm, $Vcpkg, $Ninja)
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$build = Join-Path $root 'build/release'
New-Item -ItemType Directory -Force -Path $build | Out-Null
@{Preset=$Preset; Target=$Target; Jobs=$Jobs; Rocm=$Rocm; Vcpkg=$Vcpkg; Ninja=$Ninja} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $build 'arguments.json')
if ($env:UPDATE_BUILD_EXIT -ne '0') { exit ([int]$env:UPDATE_BUILD_EXIT) }
Copy-Item -LiteralPath $env:UPDATE_ENGINE_STUB -Destination (Join-Path $build 'gufo.exe')
''')
        (self.checkout / ".gitignore").write_text("build/\n")
        (self.checkout / "source.txt").write_text("initial\n")
        self.git("init", "--initial-branch=windows-port")
        self.configure(self.checkout)
        self.git("add", ".")
        self.git("commit", "-m", "initial fixture")
        self.remote = self.root / "remote.git"
        self.git("init", "--bare", "--initial-branch=windows-port", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-u", "origin", "windows-port")
        self.git("push", "origin", "HEAD:windows-main")
        self.initial = self.git("rev-parse", "HEAD")
        self.writer = self.root / "writer"
        self.git("clone", str(self.remote), str(self.writer))
        self.configure(self.writer)
        self.git("checkout", "windows-main", cwd=self.writer)
        self.rocm = self.root / "saved rock"
        self.vcpkg = self.root / "saved vcpkg"
        self.vcpkg.mkdir()
        (self.vcpkg / "vcpkg.exe").touch()
        self.ninja = self.root / "saved ninja/ninja.exe"
        self.build_arguments = self.checkout / "build/release/arguments.json"

    def git(self, *args, cwd=None):
        result = subprocess.run([self.git_exe, "-C", str(cwd or self.checkout), *args],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def configure(self, path):
        for key, value in (("user.name", "Update fixture"), ("user.email", "fixture@example.invalid"),
                           ("commit.gpgsign", "false"), ("core.autocrlf", "false")):
            self.git("config", key, value, cwd=path)

    def advance_remote(self):
        (self.writer / "source.txt").write_text("updated\n")
        self.git("add", "source.txt", cwd=self.writer)
        self.git("commit", "-m", "remote update", cwd=self.writer)
        self.git("push", cwd=self.writer)
        return self.git("rev-parse", "HEAD", cwd=self.writer)

    def run_update(self, process=""):
        harness = self.root / "run.ps1"
        harness.write_text(
            f". {quote(self.checkout / 'tools/windows/update.ps1')} -Jobs 3\n"
            "function Get-SetupTools { [pscustomobject]@{\n"
            f"Git={quote(self.git_exe)}; VS='fixture'; CMake={quote(self.root / 'cmake.exe')};\n"
            f"Ninja={quote(self.ninja)}; Rocm={quote(self.rocm)}; Vcpkg={quote(self.vcpkg)}\n"
            "} }\n"
            f"function Get-Process {{ {process} }}\n"
            "try { Invoke-WindowsUpdate } catch { Write-Host $_.Exception.Message; exit 1 }\n",
            encoding="utf-8-sig",
        )
        return subprocess.run([self.shell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                               "-File", str(harness)], cwd=self.root, env=self.environment,
                              capture_output=True, text=True, timeout=30)

    def assert_stopped(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)
        self.assertNotIn("Update complete:", result.stdout)
        self.assertFalse(self.build_arguments.exists(), result.stdout)

    def test_fast_forward_build_paths_and_repeat_with_ignored_settings(self):
        expected = self.advance_remote()
        settings = self.checkout / "build/windows-setup.json"
        settings.parent.mkdir()
        settings.write_text('{"custom":"preserved"}')
        for _ in range(2):
            result = self.run_update()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Update complete:", result.stdout)
            self.assertEqual(self.git("rev-parse", "HEAD"), expected)
            self.assertEqual(self.git("branch", "--show-current"), "windows-main")
            self.assertEqual(self.git("rev-parse", "--abbrev-ref", "@{upstream}"),
                             "origin/windows-main")
            self.assertEqual(self.git("rev-parse", "windows-port"), self.initial)
            self.assertEqual(settings.read_text(), '{"custom":"preserved"}')
            args = json.loads(self.build_arguments.read_text(encoding="utf-8-sig"))
            self.assertEqual(args, {"Preset": "release", "Target": "gufo", "Jobs": "3",
                                    "Rocm": str(self.rocm), "Vcpkg": str(self.vcpkg),
                                    "Ninja": str(self.ninja)})

    def test_local_work_blocks_pull_and_build(self):
        self.advance_remote()
        for relative in ("source.txt", "untracked.txt"):
            with self.subTest(relative=relative):
                path = self.checkout / relative
                path.write_text("local work\n")
                self.assert_stopped(self.run_update(), "local changes or untracked files")
                self.assertEqual(self.git("rev-parse", "HEAD"), self.initial)
                self.assertEqual(path.read_text(), "local work\n")
                if relative == "source.txt":
                    self.git("restore", "source.txt")

    def test_unpublished_windows_port_commits_block_switch(self):
        self.advance_remote()
        (self.checkout / "local.txt").write_text("local commit\n")
        self.git("add", "local.txt")
        self.git("commit", "-m", "local update")
        before = self.git("rev-parse", "HEAD")
        self.git("config", "pull.rebase", "true")
        self.git("config", "rebase.autoStash", "true")
        self.assert_stopped(self.run_update(), "unpublished commits")
        self.assertEqual(self.git("rev-parse", "HEAD"), before)
        self.assertEqual(self.git("branch", "--show-current"), "windows-port")

    def test_switch_handles_divergent_remote_branch_histories(self):
        self.git("checkout", "windows-port", cwd=self.writer)
        (self.writer / "source.txt").write_text("old branch only\n")
        self.git("add", "source.txt", cwd=self.writer)
        self.git("commit", "-m", "published old branch update", cwd=self.writer)
        self.git("push", cwd=self.writer)
        old_tip = self.git("rev-parse", "HEAD", cwd=self.writer)
        self.git("fetch", "origin", "windows-port")
        self.git("merge", "--ff-only", "origin/windows-port")
        self.git("checkout", "windows-main", cwd=self.writer)
        expected = self.advance_remote()
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), expected)
        self.assertEqual(self.git("rev-parse", "windows-port"), old_tip)
        self.assertEqual((self.checkout / "source.txt").read_text(), "updated\n")

    def test_existing_windows_main_fast_forwards(self):
        self.git("branch", "windows-main")
        expected = self.advance_remote()
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), expected)
        self.assertEqual(self.git("branch", "--show-current"), "windows-main")

    def test_existing_windows_main_local_commits_block_switch(self):
        self.git("checkout", "-b", "windows-main")
        (self.checkout / "local.txt").write_text("local work\n")
        self.git("add", "local.txt")
        self.git("commit", "-m", "local main work")
        local_tip = self.git("rev-parse", "HEAD")
        self.git("checkout", "windows-port")
        self.advance_remote()
        self.assert_stopped(self.run_update(), "cannot fast-forward")
        self.assertEqual(self.git("branch", "--show-current"), "windows-port")
        self.assertEqual(self.git("rev-parse", "windows-main"), local_tip)

    def test_windows_main_divergence_cannot_rebase(self):
        self.git("checkout", "-b", "windows-main")
        (self.checkout / "local.txt").write_text("local work\n")
        self.git("add", "local.txt")
        self.git("commit", "-m", "local main work")
        before = self.git("rev-parse", "HEAD")
        self.advance_remote()
        self.git("config", "pull.rebase", "true")
        self.git("config", "merge.autoStash", "true")
        self.assert_stopped(self.run_update(), "Git failed")
        self.assertEqual(self.git("rev-parse", "HEAD"), before)

    def test_unrelated_branch_is_left_untouched(self):
        self.git("checkout", "-b", "my-work")
        self.assert_stopped(self.run_update(), "branch 'my-work' is left untouched")
        self.assertEqual(self.git("branch", "--show-current"), "my-work")

    def test_missing_tracking_still_updates_from_windows_main(self):
        expected = self.advance_remote()
        self.git("branch", "--unset-upstream")
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), expected)

    def test_detached_head_stops(self):
        self.git("checkout", "--detach")
        self.assert_stopped(self.run_update(), "detached HEAD")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.initial)

    def test_fetch_failure_prevents_build(self):
        self.git("remote", "set-url", "origin", str(self.root / "missing.git"))
        self.assert_stopped(self.run_update(), "Git failed")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.initial)

    def test_running_engine_protects_checkout_and_other_copies_are_allowed(self):
        expected = self.advance_remote()
        engine = self.checkout / "build/release/gufo.exe"
        for path in (quote(engine), "$null"):
            self.assert_stopped(self.run_update(f"[pscustomobject]@{{Id=123; Path={path}}}"),
                                "Stop it in the launcher")
            self.assertEqual(self.git("rev-parse", "HEAD"), self.initial)
        other = self.root / "another checkout/build/release/gufo.exe"
        result = self.run_update(f"[pscustomobject]@{{Id=456; Path={quote(other)}}}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), expected)

    def test_build_failure_retains_update_and_can_be_retried(self):
        expected = self.advance_remote()
        self.environment["UPDATE_BUILD_EXIT"] = "7"
        result = self.run_update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Build failed", result.stdout)
        self.assertNotIn("Update complete:", result.stdout)
        self.assertEqual(self.git("rev-parse", "HEAD"), expected)
        self.environment["UPDATE_BUILD_EXIT"] = "0"
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_version_failure_does_not_report_success(self):
        self.environment["UPDATE_VERSION_EXIT"] = "9"
        result = self.run_update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed its version check", result.stdout)
        self.assertNotIn("Update complete:", result.stdout)

    def test_batch_forwards_arguments_from_another_directory_and_exit_status(self):
        batch = self.root / "batch checkout & spaces/update-windows.bat"
        script = batch.parent / "tools/windows/update.ps1"
        script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "update-windows.bat", batch)
        script.write_text('''param($Jobs, $Rocm, $Vcpkg, $Ninja)
Write-Output "JOBS=$Jobs ROCM=$Rocm VCPKG=$Vcpkg NINJA=$Ninja"
exit ([int]$env:UPDATE_BUILD_EXIT)
''')
        for code in (0, 42):
            with self.subTest(code=code):
                self.environment["UPDATE_BUILD_EXIT"] = str(code)
                command = (f'"{os.environ["COMSPEC"]}" /d /s /c ""{batch}" -Jobs 3 '
                           f'-Rocm "{self.rocm}" -Vcpkg "{self.vcpkg}" -Ninja "{self.ninja}""')
                result = subprocess.run(command, cwd=self.root, env=self.environment, input="\n",
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn(f"JOBS=3 ROCM={self.rocm} VCPKG={self.vcpkg} NINJA={self.ninja}",
                              result.stdout)


if __name__ == "__main__":
    unittest.main()
