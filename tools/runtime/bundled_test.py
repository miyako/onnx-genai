import json
from pathlib import Path
import unittest

import artifacts


ROOT = Path(__file__).resolve().parents[2]
INFO = json.loads((ROOT / "onnx-genai/Resources/bin/macOS/onnxruntime-BUILD_INFO.json").read_text())
VERSION = INFO["version"]
REVISION = INFO["source_revision"]


class BundledRuntimeTests(unittest.TestCase):
    def assert_same_file(self, first, second):
        self.assertTrue(first.is_file(), str(first))
        self.assertTrue(second.is_file(), str(second))
        self.assertEqual(artifacts.sha256(first), artifacts.sha256(second), f"{first} != {second}")

    def check_notices(self, deployed, archived):
        for name in ("BUILD_INFO.json", "LICENSE", "ThirdPartyNotices.txt"):
            self.assert_same_file(deployed / f"onnxruntime-{name}", archived / name)
        info = json.loads((archived / "BUILD_INFO.json").read_text())
        self.assertEqual(info["version"], VERSION)
        self.assertEqual(info["source_revision"], REVISION)
        return info

    def test_macos_build_and_deployment_use_the_same_runtime(self):
        archived = ROOT / "onnxruntime" / VERSION
        deployed = ROOT / "onnx-genai/Resources/bin/macOS"
        library = archived / "libonnxruntime.dylib"
        self.assert_same_file(library, deployed / library.name)
        self.assert_same_file(library, ROOT / "cpp/macOS" / library.name)
        info = self.check_notices(deployed, archived)
        self.assertEqual(info["platform"], "macos-universal")
        self.assertEqual(info["origin"], "source-build")

    def test_windows_build_and_deployment_use_matching_release_files(self):
        for arch, deployment, link in (
            ("x64", "Windows", "windows64"),
            ("arm64", "WindowsARM", "arm64"),
        ):
            with self.subTest(arch=arch):
                archived = ROOT / "onnxruntime" / VERSION / arch
                deployed = ROOT / "onnx-genai/Resources/bin" / deployment
                info = self.check_notices(deployed, archived)
                self.assertEqual(info["platform"], f"windows-{arch}")
                self.assertEqual(info["origin"], "official-release")
                for name in ("onnxruntime", "onnxruntime_providers_shared"):
                    dll = archived / f"{name}.dll"
                    self.assert_same_file(dll, deployed / dll.name)
                    self.assertEqual(artifacts.pe_architecture(dll), arch)
                    self.assert_same_file(
                        archived / f"{name}.lib", ROOT / "cpp/lib" / link / f"{name}.lib"
                    )


if __name__ == "__main__":
    unittest.main()
