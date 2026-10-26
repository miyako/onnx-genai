import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import artifacts


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(
            archive=self.root / "runtime.zip",
            output=self.root / "package",
            version="1.29.0",
            revision="a" * 40,
            arch="arm64",
        )

    def archive(self, machine=0xAA64, extra=None, revision=None):
        binary = bytearray(128)
        binary[:2] = b"MZ"
        struct.pack_into("<I", binary, 0x3C, 80)
        binary[80:84] = b"PE\0\0"
        struct.pack_into("<H", binary, 84, machine)
        files = {
            "VERSION_NUMBER": self.args.version + "\r\n",
            "GIT_COMMIT_ID": revision or self.args.revision,
            "LICENSE": "license",
            "ThirdPartyNotices.txt": "notices",
            "include/onnxruntime_c_api.h": "header",
            "lib/onnxruntime.dll": binary,
            "lib/onnxruntime.lib": "import library",
            "lib/onnxruntime.pdb": "debug symbols",
            "lib/onnxruntime_providers_shared.dll": binary,
            "lib/onnxruntime_providers_shared.lib": "import library",
        }
        if extra:
            files.update(extra)
        with zipfile.ZipFile(self.args.archive, "w") as archive:
            for name, value in files.items():
                archive.writestr(f"onnxruntime-win-arm64-1.29.0/{name}", value)
        self.args.sha256 = artifacts.sha256(self.args.archive)

    def test_packaging_preserves_headers_and_notices_without_symbols(self):
        self.archive()
        artifacts.package_windows(self.args)
        self.assertTrue((self.args.output / "include/onnxruntime_c_api.h").is_file())
        self.assertTrue((self.args.output / "ThirdPartyNotices.txt").is_file())
        self.assertFalse((self.args.output / "lib/onnxruntime.pdb").exists())
        self.assertEqual(artifacts.pe_architecture(self.args.output / "lib/onnxruntime.dll"), "arm64")

    def test_rejects_checksum_mismatch_before_extracting(self):
        self.archive()
        self.args.sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            artifacts.package_windows(self.args)
        self.assertFalse(self.args.output.exists())

    def test_rejects_revision_mismatch(self):
        self.archive(revision="b" * 40)
        with self.assertRaisesRegex(ValueError, "GIT_COMMIT_ID"):
            artifacts.package_windows(self.args)

    def test_rejects_wrong_architecture(self):
        self.archive(machine=0x8664)
        with self.assertRaisesRegex(ValueError, "expected arm64, found x64"):
            artifacts.package_windows(self.args)

    def test_rejects_archive_traversal(self):
        self.archive(extra={"../outside.txt": "unexpected"})
        with self.assertRaisesRegex(ValueError, "archive path"):
            artifacts.package_windows(self.args)
        self.assertFalse((self.root / "outside.txt").exists())

    def test_normalizes_native_python_architectures(self):
        for machine, expected in (("AMD64", "x64"), ("ARM64", "arm64"), ("aarch64", "arm64"), ("x86_64", "x86_64")):
            with self.subTest(machine=machine), patch("artifacts.platform.machine", return_value=machine):
                self.assertEqual(artifacts.native_architecture(), expected)

    def test_manifest_and_checksums(self):
        self.archive()
        artifacts.package_windows(self.args)
        self.args.platform = "windows-arm64"
        self.args.upstream_sha256 = self.args.sha256
        artifacts.seal(self.args)
        info = json.loads((self.args.output / "BUILD_INFO.json").read_text())
        self.assertEqual(info["origin"], "official-release")
        self.assertEqual(info["source_revision"], self.args.revision)
        for line in (self.args.output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(digest, hashlib.sha256((self.args.output / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
