"""Package and smoke-check ONNX Runtime libraries without a Python ORT wheel."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import struct
import subprocess
import zipfile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pe_architecture(path):
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"Not a PE binary: {path}")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        if stream.read(4) != b"PE\0\0":
            raise ValueError(f"Invalid PE signature: {path}")
        machine = struct.unpack("<H", stream.read(2))[0]
    architectures = {0x8664: "x64", 0xAA64: "arm64"}
    if machine not in architectures:
        raise ValueError(f"Unexpected PE machine {machine:#x}: {path}")
    return architectures[machine]


def native_architecture():
    machine = platform.machine().lower()
    return {"amd64": "x64", "aarch64": "arm64"}.get(machine, machine)


def package_windows(args):
    if sha256(args.archive) != args.sha256:
        raise ValueError("Windows archive SHA-256 does not match the release digest")
    root = f"onnxruntime-win-{args.arch}-{args.version}"
    with zipfile.ZipFile(args.archive) as archive:
        for name, expected in (
            ("VERSION_NUMBER", args.version),
            ("GIT_COMMIT_ID", args.revision),
        ):
            actual = archive.read(f"{root}/{name}").decode("utf-8").strip()
            if actual != expected:
                raise ValueError(f"{name}: expected {expected}, found {actual}")
        args.output.mkdir(parents=True, exist_ok=False)
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if not parts or parts[0] != root or ".." in parts or "\\" in entry.filename:
                raise ValueError(f"Unexpected archive path: {entry.filename}")
            relative = PurePosixPath(*parts[1:])
            if entry.is_dir() or relative.suffix == ".pdb":
                continue
            target = args.output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    for name in (
        "lib/onnxruntime.dll",
        "lib/onnxruntime.lib",
        "lib/onnxruntime_providers_shared.dll",
        "lib/onnxruntime_providers_shared.lib",
        "include/onnxruntime_c_api.h",
        "LICENSE",
        "ThirdPartyNotices.txt",
    ):
        if not (args.output / name).is_file():
            raise ValueError(f"Missing required package file: {name}")
    for library in (args.output / "lib").glob("*.dll"):
        actual = pe_architecture(library)
        if actual != args.arch:
            raise ValueError(f"{library}: expected {args.arch}, found {actual}")


def verify(args):
    library = args.library.resolve(strict=True)
    if library.suffix == ".dll":
        actual = pe_architecture(library)
        if actual != args.arch:
            raise ValueError(f"Expected {args.arch}, found {actual}")
        if os.name != "nt":
            raise ValueError("DLL load checks must run on Windows")
    else:
        actual = set(subprocess.check_output(["lipo", "-archs", str(library)], text=True).split())
        expected = {"arm64", "x86_64"} if args.arch == "universal" else {args.arch}
        if actual != expected:
            raise ValueError(f"Expected architectures {expected}, found {actual}")
    native_arch = native_architecture()
    if args.arch != "universal" and native_arch != args.arch:
        raise ValueError(f"Load check needs native {args.arch} Python, found {native_arch}")

    get_api = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
    get_version = ctypes.CFUNCTYPE(ctypes.c_char_p)

    class OrtApiBase(ctypes.Structure):
        _fields_ = [("GetApi", get_api), ("GetVersionString", get_version)]

    directory = os.add_dll_directory(str(library.parent)) if os.name == "nt" else None
    try:
        runtime = ctypes.CDLL(str(library))
        runtime.OrtGetApiBase.restype = ctypes.POINTER(OrtApiBase)
        base = runtime.OrtGetApiBase().contents
        version = base.GetVersionString().decode("utf-8")
        if version != args.version:
            raise ValueError(f"Expected runtime {args.version}, found {version}")
        # The application's checked-in headers request API 23.
        if not base.GetApi(23):
            raise ValueError("Runtime does not expose the application's C API version 23")
    finally:
        if directory is not None:
            directory.close()
    print(f"Loaded ONNX Runtime {version}: {args.arch}, C API 23 available")


def seal(args):
    info = {
        "version": args.version,
        "source_revision": args.revision,
        "platform": args.platform,
        "origin": "official-release" if args.upstream_sha256 else "source-build",
        "upstream_sha256": args.upstream_sha256,
    }
    (args.output / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in args.output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    checksums = "".join(f"{sha256(path)}  {path.relative_to(args.output).as_posix()}\n" for path in files)
    (args.output / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    windows = commands.add_parser("windows")
    windows.add_argument("archive", type=Path)
    windows.add_argument("output", type=Path)
    windows.add_argument("--sha256", required=True)
    windows.add_argument("--revision", required=True)
    windows.add_argument("--arch", required=True, choices=("x64", "arm64"))
    windows.set_defaults(run=package_windows)
    check = commands.add_parser("verify")
    check.add_argument("library", type=Path)
    check.add_argument("--arch", required=True, choices=("x64", "x86_64", "arm64", "universal"))
    check.set_defaults(run=verify)
    manifest = commands.add_parser("seal")
    manifest.add_argument("output", type=Path)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--platform", required=True)
    manifest.add_argument("--upstream-sha256")
    manifest.set_defaults(run=seal)
    for command in (windows, check, manifest):
        command.add_argument("--version", required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
