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
        self.assertNotIn(b"\r", (self.args.output / "BUILD_INFO.json").read_bytes())
        self.assertNotIn(b"\r", (self.args.output / "SHA256SUMS").read_bytes())
        for line in (self.args.output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(digest, hashlib.sha256((self.args.output / name).read_bytes()).hexdigest())

    def adoption_packages(self):
        packages = self.root / "packages"
        for platform_name in ("macos-universal", "windows-x64", "windows-arm64"):
            output = packages / f"onnxruntime-{self.args.version}-{platform_name}"
            (output / "lib").mkdir(parents=True)
            names = (
                ("libonnxruntime.dylib",)
                if platform_name == "macos-universal"
                else (
                    "onnxruntime.dll", "onnxruntime.lib",
                    "onnxruntime_providers_shared.dll", "onnxruntime_providers_shared.lib",
                )
            )
            for name in names:
                (output / "lib" / name).write_text(platform_name + name)
            (output / "LICENSE").write_text("license")
            (output / "ThirdPartyNotices.txt").write_text("notices")
            artifacts.seal(SimpleNamespace(
                output=output, version=self.args.version, revision=self.args.revision,
                platform=platform_name, upstream_sha256=None,
            ))
        return SimpleNamespace(
            packages=packages, root=self.root / "repo",
            version=self.args.version, revision=self.args.revision,
        )

    def test_adoption_updates_link_and_deployment_files_and_notices(self):
        args = self.adoption_packages()
        header = args.root / "cpp/include/onnxruntime_c_api.h"
        header.parent.mkdir(parents=True)
        header.write_text("keep API 23")
        artifacts.adopt(args)
        self.assertEqual(header.read_text(), "keep API 23")
        for platform_name, deployment, link, arch in (
            ("macos-universal", "macOS", "macOS", ""),
            ("windows-x64", "Windows", "lib/windows64", "x64"),
            ("windows-arm64", "WindowsARM", "lib/arm64", "arm64"),
        ):
            package = args.packages / f"onnxruntime-{args.version}-{platform_name}"
            archived = args.root / "onnxruntime" / args.version / arch
            deployed = args.root / "onnx-genai/Resources/bin" / deployment
            for source in (package / "lib").iterdir():
                self.assertEqual(source.read_bytes(), (archived / source.name).read_bytes())
                target = args.root / "cpp" / link if source.suffix == ".lib" or not arch else deployed
                self.assertEqual(source.read_bytes(), (target / source.name).read_bytes())
                if not arch:
                    self.assertEqual(source.read_bytes(), (deployed / source.name).read_bytes())
            for name in ("BUILD_INFO.json", "LICENSE", "ThirdPartyNotices.txt"):
                self.assertEqual((package / name).read_bytes(), (deployed / f"onnxruntime-{name}").read_bytes())
                self.assertEqual((package / name).read_bytes(), (archived / name).read_bytes())

    def test_adoption_validates_every_platform_before_copying(self):
        args = self.adoption_packages()
        package = args.packages / f"onnxruntime-{args.version}-windows-arm64"
        (package / "lib/onnxruntime.dll").write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            artifacts.adopt(args)
        self.assertFalse(args.root.exists())

    def test_adoption_rejects_mixed_releases(self):
        args = self.adoption_packages()
        args.revision = "b" * 40
        with self.assertRaisesRegex(ValueError, "source_revision"):
            artifacts.adopt(args)
        self.assertFalse(args.root.exists())

    def test_adoption_requires_checksums_for_deployed_libraries(self):
        args = self.adoption_packages()
        package = args.packages / f"onnxruntime-{args.version}-windows-x64"
        checksums = package / "SHA256SUMS"
        checksums.write_text("".join(
            line for line in checksums.read_text().splitlines(keepends=True)
            if "lib/onnxruntime.dll" not in line
        ))
        with self.assertRaisesRegex(ValueError, "missing required checksums"):
            artifacts.adopt(args)
        self.assertFalse(args.root.exists())

    def test_adoption_rejects_checksum_path_traversal(self):
        args = self.adoption_packages()
        package = args.packages / f"onnxruntime-{args.version}-macos-universal"
        with (package / "SHA256SUMS").open("a") as stream:
            stream.write(f"{'0' * 64}  ../outside\n")
        with self.assertRaisesRegex(ValueError, "checksum path"):
            artifacts.adopt(args)
        self.assertFalse(args.root.exists())


if __name__ == "__main__":
    unittest.main()
