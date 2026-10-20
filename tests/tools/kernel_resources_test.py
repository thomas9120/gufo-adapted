#!/usr/bin/env python3
"""GPU-independent fixtures for the prefill resource regression guard."""

import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "kernel_resources", ROOT / "tools/ci/check-kernel-resources.py")
RESOURCES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOURCES)


def pe(payload, name=b".hip_fat"):
    data = bytearray(256)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 70, 1)
    data[88:96] = name.ljust(8, b"\0")
    struct.pack_into("<II", data, 104, len(payload), len(data))
    return bytes(data) + payload


def elf(payload):
    names = b"\0.shstrtab\0.hip_fatbin\0"
    data = bytearray(256)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", data, 0x28, 64)
    struct.pack_into("<HHH", data, 0x3A, 64, 3, 1)
    struct.pack_into("<IIQQQQIIQQ", data, 128, 1, 3, 0, 0,
                     256, len(names), 0, 0, 1, 0)
    struct.pack_into("<IIQQQQIIQQ", data, 192, 11, 1, 0, 0,
                     256 + len(names), len(payload), 0, 0, 1, 0)
    return bytes(data) + names + payload


def clean_kernels():
    return {("gfx1151", prefix + "PKvPfyyy"):
            {"scratch": 0, "vgpr": 192, "vgpr_spill": 0, "sgpr_spill": 0}
            for prefix in RESOURCES.required_variants()}


class KernelResourcesTest(unittest.TestCase):
    def test_pe_and_elf_payloads(self):
        payload = RESOURCES.make_bundle([
            ("host-x86_64", b""), ("hipv4-amdgcn-amd-amdhsa--gfx1151", b"device")])
        for binary in (pe(payload), elf(payload)):
            with self.subTest(kind=binary[:2]):
                self.assertEqual(RESOURCES.fatbin_section(binary), payload)
                self.assertEqual(len(list(RESOURCES.split_bundles(payload))), 1)

    def test_invalid_or_truncated_input(self):
        for binary in (b"", b"MZ", pe(b"data", b".hipFatB"), pe(b"data")[:-1]):
            with self.subTest(binary=binary[:8]):
                with self.assertRaises((ValueError, struct.error)):
                    RESOURCES.fatbin_section(binary)
        # A zero-length CCOB used to leave the scanning cursor unchanged.
        with self.assertRaises(ValueError):
            list(RESOURCES.split_bundles(b"CCOB" + struct.pack("<HHQ", 3, 0, 0)))
        with self.assertRaises((ValueError, struct.error)):
            list(RESOURCES.split_bundles(RESOURCES.make_bundle([("hip", b"co")])[:-1]))

    def test_no_kernels_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gufo.exe"
            path.write_bytes(pe(RESOURCES.make_bundle([("host-x86_64", b"")])))
            with self.assertRaisesRegex(ValueError, "no AMDGPU kernels"):
                RESOURCES.binary_kernels(str(path), None)

    def test_uncompressed_code_object_metadata_and_duplicates(self):
        blob = RESOURCES.make_bundle([
            ("hipv4-amdgcn-amd-amdhsa--gfx1151", b"first"),
            ("hipv4-amdgcn-amd-amdhsa--gfx1151", b"second"),
            ("hipv4-amdgcn-amd-amdhsa--gfx1100", b"other")])
        def metadata(payload):
            return [{".name": "kernel", ".vgpr_count": 128,
                     ".private_segment_fixed_size": 80 if payload == b"first" else 0}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gufo.exe"
            path.write_bytes(pe(blob))
            with mock.patch.object(RESOURCES, "code_object_kernels", side_effect=metadata):
                kernels = RESOURCES.binary_kernels(str(path), None)
            self.assertEqual(len(kernels), 2)
            self.assertEqual(kernels["gfx1151", "kernel"]["scratch"], 80)
            self.assertEqual(kernels["gfx1100", "kernel"]["scratch"], 0)

    def test_amdgpu_note_metadata(self):
        def string(value):
            encoded = value.encode()
            return bytes([0xA0 + len(encoded)]) + encoded
        # A hand-authored AMDGPU note: one kernel, private array but no spills.
        metadata = (b"\x81" + string("amdhsa.kernels") + b"\x91\x83" +
                    string(".name") + string("kernel") +
                    string(".private_segment_fixed_size") + b"\x50" +
                    string(".vgpr_count") + b"\xcc\x80")
        note = struct.pack("<III", 7, len(metadata), 32) + b"AMDGPU\0\0" + metadata
        note += bytes((-len(metadata)) % 4)
        binary = bytearray(elf(note))
        # Convert the payload section from PROGBITS to SHT_NOTE.
        struct.pack_into("<I", binary, 196, 7)
        self.assertEqual(RESOURCES.code_object_kernels(bytes(binary)), [{
            ".name": "kernel", ".private_segment_fixed_size": 80, ".vgpr_count": 128}])

    def test_bundler_gets_one_output_per_target(self):
        targets = ["host-x86_64", "hipv4-amdgcn-amd-amdhsa--gfx1151"]
        payloads = [b"", b"device code"]
        blob = b"CCOB" + struct.pack("<HH", 3, 65535) + bytes(24)
        def bundler(command, **kwargs):
            if "--list" in command:
                return subprocess.CompletedProcess(command, 0, "\n".join(targets))
            outputs = [arg.removeprefix("--output=") for arg in command
                       if arg.startswith("--output=")]
            self.assertEqual(len(outputs), len(targets))
            self.assertIn("--targets=" + ",".join(targets), command)
            for output, payload in zip(outputs, payloads):
                Path(output).write_bytes(payload)
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(RESOURCES.subprocess, "run", side_effect=bundler):
            result = RESOURCES.inflate_ccob(blob, "/hip/clang-offload-bundler.exe")
        self.assertEqual(list(RESOURCES.bundle_entries(result)), list(zip(targets, payloads)))

    def test_zlib_compressed_payload(self):
        import zlib
        payload = RESOURCES.make_bundle([("hipv4-amdgcn-amd-amdhsa--gfx1151", b"co")])
        header = b"CCOB" + struct.pack("<HH", 3, 0) + bytes(24)
        self.assertEqual(RESOURCES.inflate_ccob(header + zlib.compress(payload), None), payload)

    def test_complete_zero_scratch_contract(self):
        self.assertEqual(len(RESOURCES.required_variants()), 70)
        self.assertEqual(RESOURCES.check(clean_kernels()), [])
        # The host argument ABI is outside the required template prefix.
        linux = {(arch, name[:-3] + "mmm"): resource
                 for (arch, name), resource in clean_kernels().items()}
        self.assertEqual(RESOURCES.check(linux), [])

    def test_missing_variant_or_wrong_architecture_fails(self):
        kernels = clean_kernels()
        removed = next(k for k in kernels if "Lb1ELb1" in k[1])
        resource = kernels.pop(removed)
        kernels["gfx1100", removed[1]] = resource
        errors = RESOURCES.check(kernels)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing required gfx1151 variant", errors[0])
        self.assertTrue(RESOURCES.check({}))

    def test_private_array_and_fused_spill_fail(self):
        kernels = clean_kernels()
        paired = next(k for k in kernels if "RoutedF16" in k[1] and "Lb1" in k[1])
        fused = next(k for k in kernels if "Lb1ELb1" in k[1])
        kernels[paired]["scratch"] = 80
        kernels[fused]["vgpr_spill"] = 1
        self.assertEqual(len(RESOURCES.check(kernels)), 2)


if __name__ == "__main__":
    unittest.main()
