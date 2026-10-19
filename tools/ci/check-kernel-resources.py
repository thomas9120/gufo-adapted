#!/usr/bin/env python3
"""Reject scratch/spills in the gfx1151 prefill GEMMs affected by PR #459.

Reads Windows PE or Linux ELF HIP binaries without a GPU. Offload and AMDGPU
metadata decoding is adapted from gufo-org/gufo PR #459 (head 9bccebc5).
The required template prefixes exclude host argument ABI encodings, so the
same focused zero-scratch contract applies on Windows and Linux.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
CCOB_MAGIC = b"CCOB"
NT_AMDGPU_METADATA = 32


# ---- ELF --------------------------------------------------------------------------------
def elf_sections(data: bytes) -> dict[str, tuple[int, int, int]]:
    """Return name -> (type, offset, size) for each section of an ELF64 LE file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def header(i: int) -> tuple:
        return struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)

    strtab = header(shstrndx)[4]
    out = {}
    for i in range(shnum):
        name_off, sh_type, _, _, off, size = header(i)[:6]
        end = data.index(b"\0", strtab + name_off)
        out[data[strtab + name_off:end].decode()] = (sh_type, off, size)
    return out


def fatbin_section(data: bytes) -> bytes:
    """Locate the payload, not Windows' separate .hipFatB wrapper section."""
    if data.startswith(b"\x7fELF"):
        _, offset, size = elf_sections(data)[".hip_fatbin"]
    elif data.startswith(b"MZ"):
        pe, = struct.unpack_from("<I", data, 0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("invalid PE signature")
        count, = struct.unpack_from("<H", data, pe + 6)
        optional_size, = struct.unpack_from("<H", data, pe + 20)
        table = pe + 24 + optional_size
        if table + count * 40 > len(data):
            raise ValueError("truncated PE section table")
        for i in range(count):
            entry = table + i * 40
            if data[entry:entry + 8].rstrip(b"\0") == b".hip_fat":
                size, offset = struct.unpack_from("<II", data, entry + 16)
                break
        else:
            raise ValueError("no .hip_fat section")
    else:
        raise ValueError("expected a PE or ELF HIP binary")
    if size == 0 or offset + size > len(data):
        raise ValueError("empty or truncated HIP payload section")
    return data[offset:offset + size]


def elf_notes(data: bytes):
    """Yield (type, name, desc) for each note in each SHT_NOTE section."""
    for sh_type, off, size in elf_sections(data).values():
        if sh_type != 7:  # SHT_NOTE
            continue
        pos, end = off, off + size
        while pos + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", data, pos)
            pos += 12
            name = data[pos:pos + namesz].rstrip(b"\0")
            pos += (namesz + 3) & ~3
            desc = data[pos:pos + descsz]
            pos += (descsz + 3) & ~3
            yield ntype, name, desc


# ---- msgpack (only the part that the AMDGPU metadata uses) ------------------------------
def unpack(buf: bytes, pos: int = 0):
    b = buf[pos]
    pos += 1
    if b <= 0x7F:
        return b, pos
    if 0x80 <= b <= 0x8F:
        return unpack_map(buf, pos, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return unpack_array(buf, pos, b & 0x0F)
    if 0xA0 <= b <= 0xBF:
        n = b & 0x1F
        return buf[pos:pos + n].decode(errors="replace"), pos + n
    if b >= 0xE0:
        return b - 0x100, pos
    fixed = {0xC0: None, 0xC2: False, 0xC3: True}
    if b in fixed:
        return fixed[b], pos
    fmt = {0xCC: "B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: "b", 0xD1: ">h",
           0xD2: ">i", 0xD3: ">q", 0xCA: ">f", 0xCB: ">d"}
    if b in fmt:
        v, = struct.unpack_from(fmt[b], buf, pos)
        return v, pos + struct.calcsize(fmt[b])
    if b in (0xD9, 0xDA, 0xDB, 0xC4, 0xC5, 0xC6):
        width = {0xD9: 1, 0xDA: 2, 0xDB: 4, 0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
        n = int.from_bytes(buf[pos:pos + width], "big")
        pos += width
        raw = buf[pos:pos + n]
        return (raw.decode(errors="replace") if b >= 0xD9 else raw), pos + n
    if b in (0xDC, 0xDD):
        width = 2 if b == 0xDC else 4
        return unpack_array(buf, pos + width, int.from_bytes(buf[pos:pos + width], "big"))
    if b in (0xDE, 0xDF):
        width = 2 if b == 0xDE else 4
        return unpack_map(buf, pos + width, int.from_bytes(buf[pos:pos + width], "big"))
    raise ValueError(f"unsupported msgpack byte 0x{b:02x}")


def unpack_array(buf, pos, n):
    out = []
    for _ in range(n):
        v, pos = unpack(buf, pos)
        out.append(v)
    return out, pos


def unpack_map(buf, pos, n):
    out = {}
    for _ in range(n):
        k, pos = unpack(buf, pos)
        v, pos = unpack(buf, pos)
        out[k] = v
    return out, pos


def code_object_kernels(co: bytes) -> list[dict]:
    for ntype, name, desc in elf_notes(co):
        if ntype == NT_AMDGPU_METADATA and name == b"AMDGPU":
            meta, _ = unpack(desc)
            return meta.get("amdhsa.kernels", [])
    return []


# ---- offload bundles --------------------------------------------------------------------
def split_bundles(fatbin: bytes):
    """Yield each offload bundle (uncompressed or CCOB) in a .hip_fatbin section."""
    pos = 0
    while True:
        cands = [p for p in (fatbin.find(BUNDLE_MAGIC, pos), fatbin.find(CCOB_MAGIC, pos)) if p >= 0]
        if not cands:
            return
        p = min(cands)
        if fatbin.startswith(BUNDLE_MAGIC, p):
            count, = struct.unpack_from("<Q", fatbin, p + 24)
            q, end = p + 32, p
            for _ in range(count):
                off, size, tlen = struct.unpack_from("<QQQ", fatbin, q)
                q += 24 + tlen
                if q > len(fatbin) or p + off + size > len(fatbin):
                    raise ValueError("truncated offload bundle")
                end = max(end, p + off + size)
            if end < q:
                raise ValueError("invalid offload bundle size")
            yield fatbin[p:end]
            pos = end
        else:
            version, = struct.unpack_from("<H", fatbin, p + 4)
            total = struct.unpack_from("<Q" if version >= 3 else "<I", fatbin, p + 8)[0]
            if total < (32 if version >= 3 else 24) or p + total > len(fatbin):
                raise ValueError("invalid compressed offload bundle size")
            yield fatbin[p:p + total]
            pos = p + total


def bundle_entries(bundle: bytes):
    """Yield (triple, payload) for each entry of an uncompressed offload bundle."""
    count, = struct.unpack_from("<Q", bundle, 24)
    q = 32
    for _ in range(count):
        off, size, tlen = struct.unpack_from("<QQQ", bundle, q)
        triple = bundle[q + 24:q + 24 + tlen].decode()
        q += 24 + tlen
        yield triple, bundle[off:off + size]


def inflate_ccob(blob: bytes, bundler: str | None) -> bytes:
    version, method = struct.unpack_from("<HH", blob, 4)
    header = 32 if version >= 3 else 24
    payload = blob[header:]
    try:
        if method == 0:  # llvm::compression::Format::Zlib
            import zlib
            return zlib.decompress(payload)
        if method == 1:  # llvm::compression::Format::Zstd
            try:
                from compression import zstd  # Python 3.14+
            except ImportError:
                import zstandard as zstd  # type: ignore
            return zstd.decompress(payload)
    except Exception:
        pass
    if not bundler:
        raise RuntimeError("compressed offload bundle: needs Python zstd or --bundler")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in")
        Path(src).write_bytes(blob)
        targets = subprocess.run([bundler, "--list", "--type=o", f"--input={src}"],
                                 check=True, capture_output=True, text=True,
                                 timeout=30).stdout.split()
        if not targets:
            raise ValueError("compressed bundle has no targets")
        outs = [os.path.join(tmp, f"t{i}") for i in range(len(targets))]
        subprocess.run([bundler, "--unbundle", "--type=o", f"--input={src}",
                        "--targets=" + ",".join(targets),
                        *[f"--output={path}" for path in outs]],
                       check=True, capture_output=True, timeout=30)
        # Make an uncompressed bundle again. Then one code path reads the two types.
        entries = [(t, Path(o).read_bytes()) for t, o in zip(targets, outs)]
        return make_bundle(entries)


def make_bundle(entries):
    head = BUNDLE_MAGIC + struct.pack("<Q", len(entries))
    table_len = sum(24 + len(t.encode()) for t, _ in entries)
    off = len(head) + table_len
    table, body = b"", b""
    for t, payload in entries:
        table += struct.pack("<QQQ", off + len(body), len(payload), len(t.encode())) + t.encode()
        body += payload
    return head + table + body


QWEN_W8 = "_ZN4gufo3hip25W8A8BlockedWmmaGEMMKernelI"
QWEN_K = "_ZN4gufo3hip30WKQuantA8BlockedWmmaGEMMKernelI"
FLASH = "_ZN4gufo6models17qwen38_flash_next4rocm12_GLOBAL__N_1"
FLASH_W8 = FLASH + "25W8A8BlockedWmmaGEMMKernelI"
FLASH_ROUTED = FLASH + "19RoutedF16GEMMKernelI"
FAMILIES = (QWEN_W8, QWEN_K, FLASH_W8, FLASH_ROUTED)


def required_variants() -> list[str]:
    """Pinned production dispatch coverage, independent of size_t's host ABI."""
    tiles = ((128, 32, 4, 4, 2), (128, 16, 4, 8, 1),
             (128, 128, 2, 4, 2), (128, 64, 4, 8, 1))

    def ints(values):
        return "".join(f"Li{v}E" for v in values)

    names = [QWEN_W8 + ints(t) + "Lb0ELb0EEEv" for t in tiles]
    names.append(QWEN_W8 + ints(tiles[2]) + "Lb1ELb1EEEv")
    for fmt in (12, 13, 14, 11, 20, 23, 21):
        for tile in tiles:
            names.append(QWEN_K + ints(tile) + f"LNS_4core8GgmlTypeE{fmt}ELi32EEEv")
    for fmt, wm, wn in ((12, 4, 1), (13, 4, 1), (14, 4, 1),
                        (8, 2, 2), (23, 2, 2), (8, 4, 1)):
        names.append(QWEN_K + ints((128, 128, 2, wm, wn)) +
                     f"LNS_4core8GgmlTypeE{fmt}ELi64EEEv")
    for tile, half in (((64, 128, 4, 2, 4), 1), ((64, 128, 4, 2, 4), 0),
                       ((128, 128, 2, 4, 2), 0), ((128, 64, 4, 4, 2), 0)):
        names.append(FLASH_W8 + ints(tile) + f"Lb{half}EEEv")
    for fmt in (12, 7, 8, 13, 20, 21, 23):
        for bn in (16, 48):
            names.append(FLASH_ROUTED + f"LNS2_10WeightTypeE{fmt}E" +
                         ints((128, bn, 2)) + "Lb0EEEv")
    for fmt in (7, 8, 20, 21, 23):
        names.append(FLASH_ROUTED + f"LNS2_10WeightTypeE{fmt}E" +
                     ints((128, 64, 2)) + "Lb0EEEv")
    for fmt in (12, 13):
        for bn in (64, 128):
            names.append(FLASH_ROUTED + f"LNS2_10WeightTypeE{fmt}E" +
                         ints((128, bn, 2)) + "Lb1EEEv")
    return names


def binary_kernels(path: str, bundler: str | None) -> dict[tuple[str, str], dict]:
    kernels = {}
    for blob in split_bundles(fatbin_section(Path(path).read_bytes())):
        if blob.startswith(CCOB_MAGIC):
            blob = inflate_ccob(blob, bundler)
        for triple, payload in bundle_entries(blob):
            if not triple.startswith("hip") or "amdgcn" not in triple or not payload:
                continue
            arch = triple.rsplit("--", 1)[-1].split(":", 1)[0]
            for k in code_object_kernels(payload):
                # Missing resource fields must not masquerade as zero scratch.
                resources = {"scratch": int(k[".private_segment_fixed_size"]),
                             "vgpr": int(k[".vgpr_count"]),
                             "vgpr_spill": int(k.get(".vgpr_spill_count", 0)),
                             "sgpr_spill": int(k.get(".sgpr_spill_count", 0))}
                identity = (arch, k[".name"])
                if identity in kernels:
                    resources = {field: max(value, kernels[identity][field])
                                 for field, value in resources.items()}
                kernels[identity] = resources
    if not kernels:
        raise ValueError("no AMDGPU kernels found")
    return kernels


def check(kernels: dict[tuple[str, str], dict]) -> list[str]:
    selected = {name: resource for (arch, name), resource in kernels.items()
                if arch == "gfx1151" and name.startswith(FAMILIES)}
    errors = []
    for prefix in required_variants():
        if not any(name.startswith(prefix) for name in selected):
            errors.append(f"missing required gfx1151 variant: {prefix}")
    for name, resource in sorted(selected.items()):
        if any(resource[field] != 0 for field in ("scratch", "vgpr_spill", "sgpr_spill")):
            errors.append(f"{name}: {resource['scratch']} B scratch, "
                          f"{resource['vgpr_spill']} VGPR spills, "
                          f"{resource['sgpr_spill']} SGPR spills")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary")
    parser.add_argument("--bundler", help="matching clang-offload-bundler executable")
    parser.add_argument("--report", action="store_true", help="report affected kernel resources")
    args = parser.parse_args()
    try:
        if args.bundler and not Path(args.bundler).is_file():
            raise ValueError(f"offload bundler not found: {args.bundler}")
        kernels = binary_kernels(args.binary, args.bundler)
        errors = check(kernels)
        selected = [(name, r) for (arch, name), r in kernels.items()
                    if arch == "gfx1151" and name.startswith(FAMILIES)]
        if args.report:
            for name, r in sorted(selected):
                print(f"{r['scratch']:4d} B scratch, {r['vgpr']:3d} VGPRs, "
                      f"{r['vgpr_spill']:3d}/{r['sgpr_spill']:3d} VGPR/SGPR spills: {name}")
        for error in errors:
            print("FAIL " + error)
        print(f"{len(selected)} affected gfx1151 kernels checked; {len(errors)} failures")
        return 1 if errors else 0
    except (OSError, ValueError, KeyError, IndexError, struct.error,
            RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
