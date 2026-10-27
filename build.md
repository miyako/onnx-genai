---
layout: default
---

# Memo: Build Instructions

## Automated ONNX Runtime libraries

The C++ link dependencies and deployed ONNX Runtime libraries currently use
**1.29.0** on macOS universal, Windows x64 and Windows ARM64. The workflow below
produces replacement artifacts, commits them to the selected branch, and then
rebuilds the server CLI against those exact dependencies.
The macOS Xcode target matches the runtime's 13.3 deployment floor. Other
dependencies may impose higher requirements.

The initial bundled upgrade comes from
[workflow run 40365235948](https://github.com/miyako/onnx-genai/actions/runs/40365235948),
which completed both native macOS source builds, universal loading checks on
both architectures, and native Windows x64/ARM64 loading checks. All packages
use upstream source revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.
The deployed folders include the runtime's license, third-party notices and
`onnxruntime-BUILD_INFO.json`. Application headers remain at C API 23, which the
new runtime exposes for backward compatibility.

Once the workflow is on the repository's default branch, run
**Actions > ONNX Runtime libraries > Run workflow** and enter an official
stable version without the `v` prefix. The default is **1.29.0**, which has been
verified with the d1 adapter. A successful run now upgrades the runtime
bundled with the application and starts the server CLI build automatically.
Dispatch on a branch, not a tag, to enable the update/build jobs.

The workflow `.github/workflows/onnxruntime.yml` produces:

| Artifact | Origin | Contents |
| --- | --- | --- |
| `onnxruntime-<version>-macos-universal` | Source build, pinned to the release commit | Developer ID signed universal `lib/libonnxruntime.dylib`, notarized/stapled DMG, notarization result, C/C++ headers, licenses |
| `onnxruntime-<version>-windows-x64` | Official CPU release archive | x64 DLLs/import libraries in `lib/`, headers, licenses |
| `onnxruntime-<version>-windows-arm64` | Official CPU release archive | ARM64 DLLs/import libraries in `lib/`, headers, licenses |

macOS release assets currently cover ARM64, not a universal library. Each macOS
slice is built on its native runner (`macos-15` and `macos-15-intel`) before
combining them with `lipo`. CPU/contrib kernels are retained; this is not a
minimal or GPU build. The deployment target is **macOS 13.3**, matching the
current upstream baseline, rather than the older application project setting.
Builds do not use `-march=native`, do not unlink Homebrew packages, and exclude
Homebrew prefixes from CMake dependency searches.

Windows uses the existing upstream binaries instead of recompiling identical
libraries. Both downloads must match the release's published SHA-256 digest,
version, source revision, and PE architecture. The workflow deliberately fails
if a requested release lacks either Windows archive or its published digest.
PDBs are omitted from the resulting packages.

The final macOS library is checked for both architecture slices and loaded on
both Apple Silicon and Intel. Windows libraries are loaded on native x64 and
ARM64 runners (`windows-2022` and `windows-11-arm`). The checks verify the runtime
version and availability of C API 23, used by this application's headers;
they are library-loading checks, not full model or GenAI compatibility tests.
Each final artifact includes `BUILD_INFO.json` and `SHA256SUMS`.

The final universal dylib is **Developer ID signed** with hardened runtime and
a secure timestamp on a hosted runner, using the same secrets as `Publish`:
`APPLE_DEVELOPER_ID_CERTIFICATE`, `APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD`,
`KEYCHAIN_PASSWORD`, `CODESIGN_APPLE_ID`, `NOTARYTOOL_APPLE_ID`,
`NOTARYTOOL_TEAM_ID`, and `NOTARYTOOL_PASSWORD`.
The workflow imports the certificate into a temporary keychain, signs the dylib,
then submits a signed `libonnxruntime.dylib.dmg` to Apple. It requires an
`Accepted` result, staples and validates the DMG's ticket, and checks Gatekeeper
assessment before sealing the package. Credentials are removed even on failure;
there is no ad-hoc fallback if signing or notarization fails.

A bare dylib cannot carry a stapled ticket. The signed library is notarized
inside the DMG; the DMG provides the stapled ticket for offline distribution.
The workflow preserves that DMG and `NOTARIZATION.json` in both the artifact and
`onnxruntime/<version>/`, while deploying the exact signed dylib to the server's
build/runtime directories. The native intermediate slices remain ad-hoc signed.

The artifact helpers use only Python's standard library (Python 3.11+).
Run their focused checks with:

```sh
python3 -m unittest discover -s tools/runtime -p '*_test.py'
actionlint .github/workflows/onnxruntime.yml .github/workflows/server-cli.yml
```

After all native checks succeed, the workflow verifies the three packages'
checksums, version, and source revision before replacing the deployed runtime
libraries, `cpp/macOS/libonnxruntime.dylib`, Windows import libraries under
`cpp/lib/`, and the corresponding `onnxruntime/<version>/` archive.
Licenses and build metadata are updated together with the libraries.
Application headers stay at C API 23; they are not replaced with upstream headers.

The runtime update is committed with `GITHUB_TOKEN`, then the workflow explicitly
calls `.github/workflows/server-cli.yml` with that commit and branch. This does
not rely on a bot push triggering another workflow (GitHub suppresses those
push-triggered runs). Repository Actions must be allowed to write contents;
branch protection must permit these bot updates. Pushes are fast-forward only:
if another change lands during a build, the update fails rather than overwriting
it. Dispatch a new run from the current branch head in that case.

### Automated server CLI

The **Server CLI** workflow also runs on `main` pushes affecting `cpp/`, the
runtime helpers, or its workflow file, and supports manual dispatch using the
currently bundled dependencies. It builds the existing Xcode and Visual Studio
projects in Release mode: macOS universal (arm64 + x86_64), Windows x64, and
Windows ARM64. CLI builds/checks use `macos-26` and `macos-26-intel` because the
existing Extensions dylib and some static dependencies were built for macOS 26;
the runtime-only jobs retain their macOS 15 runners. The CLI's 13.3 project
deployment setting does not lower those dependencies' OS requirements.
The macOS build overrides `-march=native` so both slices are
portable and disables Developer ID signing; the packaged executable is ad-hoc
signed instead. Windows uses the architecture-specific import libraries.

Each build stages the rebuilt executable alongside the checked-in runtime and
other dynamic dependencies, then runs the model-free HTTP/CLI regression checks.
The universal macOS package is exercised on both Apple Silicon and Intel;
Windows packages are exercised on native x64 and ARM64 runners. After all checks
succeed, the workflow commits the rebuilt executables to
`onnx-genai/Resources/bin/{macOS,Windows,WindowsARM}/` on the same branch.
Complete server packages are also uploaded as `server-cli-macos-universal`,
`server-cli-windows-x64`, and `server-cli-windows-arm64`, retained for 30 days.
After downloading a macOS artifact, restore the executable bit with
`chmod +x onnx-genai` (GitHub artifact downloads do not preserve permissions).

The 4D component workflow remains unchanged: it runs on its existing 4D-source
and project-resource paths or manual dispatch, not these binary updates.
Neither native workflow publishes a release. A runtime update is committed
before the CLI build; if compilation or regression checks fail, the old CLI
remains checked in and the workflow fails visibly. Fix the failure and dispatch
**Server CLI** on the updated branch before releasing a component.

These workflows do not rebuild ONNX Runtime GenAI, Extensions, or the other
static dependencies; their existing versions are intentionally left unchanged.
The regression checks do not replace real-model compatibility testing: exercise
the models you intend to use before publishing a 4D component.
Only the ONNX Runtime dylib is Developer ID signed and notarized by the runtime
workflow. The server CLI executable remains ad-hoc signed, and the existing
GenAI/Extensions libraries are not re-signed by these workflows.
Native Windows ARM64 runner availability depends on the repository's GitHub
Actions plan.

Sources: [upstream CPU build instructions](https://onnxruntime.ai/docs/build/inferencing.html),
[official releases](https://github.com/microsoft/onnxruntime/releases), and
[GitHub runner images](https://github.com/actions/runner-images#available-images).

## Manual dependency builds

### `onnxruntime`

```
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

```
brew unlink protobuf
brew unlink abseil
./build.sh \
  --config Release \
  --build_shared_lib \
  --parallel \
  --osx_arch arm64 \
  --build_dir build/arm64 \
  --skip_tests
brew link abseil
brew link protobuf
```

```
brew unlink protobuf
brew unlink abseil
./build.sh \
  --config Release \
  --build_shared_lib \
  --parallel \
  --osx_arch x86_64 \
  --build_dir build/x64 \
  --skip_tests
brew link abseil
brew link protobuf
```

```
.\build.bat ^
  --config Release ^
  --build_shared_lib ^
  --parallel ^
  --skip_tests ^
  --cmake_generator "Visual Studio 17 2022"
```

```
.\build.bat ^
  --config Release ^
  --build_shared_lib ^
  --parallel ^
  --skip_tests ^
  --cmake_generator "Visual Studio 17 2022" --arm64
```

> [!TIP]
> Official Windows CPU packages avoid rebuilding ONNX Runtime. The macOS
> ONNX Runtime release package is ARM64-only; use source builds for a universal
> dylib. ONNX Runtime GenAI is a separate dependency.

### `onnxruntime-extensions`

```
git clone --recursive https://github.com/microsoft/onnxruntime-extensions.git
cd onnxruntime-extensions
```

```
mkdir build_universal
cd build_universal
cmake .. \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOCOS_ENABLE_OPENCV_CODECS=OFF
cmake --build . --config Release --parallel
```

```
cmake -S . -B build -G "Visual Studio 17 2022" -A ARM64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOCOS_BUILD_SHARED_LIB=ON
cmake --build . --config Release --parallel
```

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOCOS_BUILD_SHARED_LIB=ON
cmake --build . --config Release --parallel
```

### `onnxruntime-genai`

```
git clone --recursive https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
```

```
python3 -m venv ort
source ort/bin/activate
pip3 install requests numpy
```

```
python3 build.py \
  --build_dir build_arm64 \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 \
  --skip_tests
```

```
python3 build.py \
  --build_dir build_x64 \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=x86_64 \
  --skip_tests
```

> There is no osx-x64 prebuilt runtime in `onnxruntime-genai/build/Release/_deps/ortlib-src/runtimes/osx-x64/native`.
> Place a copy of `onnxruntime` manually.










## ONNX Runtime Extensions

```
python -c "with open('version.txt', 'w', encoding='utf-8') as f: f.write('0.12.0')"
```

```
rmdir /s /q build
cmake -S . -B build -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DONNXRUNTIME_EXTENSION_BUILD_SHARED_LIB=OFF ^
  -DONNXRUNTIME_EXTENSION_ENABLE_PYTHON=OFF ^
  -DONNXRUNTIME_EXTENSION_ENABLE_EXAMPLES=OFF ^
  -Dprotobuf_BUILD_TESTS=OFF ^
  -Dprotobuf_BUILD_PROTOC_BINARIES=OFF ^
  -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
  -Ddlib_NO_GUI_SUPPORT=ON ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_C_COMPILER=clang-cl ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_C_COMPILER_TARGET=arm64-pc-windows-msvc ^
  -DCMAKE_SYSTEM_NAME=Windows ^
  -DCMAKE_CXX_COMPILER_TARGET=arm64-pc-windows-msvc ^
  -DOCOS_ENABLE_CTEST=OFF -DBUILD_TESTING=OFF
```

```
rmdir /s /q build
cmake -S . -B build -G "Visual Studio 17 2022" -A ARM64 -DOCOS_ENABLE_CTEST=OFF -DBUILD_TESTING=OFF ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DONNXRUNTIME_EXTENSION_ENABLE_TESTS=OFF ^
  -DONNXRUNTIME_EXTENSION_ENABLE_DOCS=OFF ^
  -Dprotobuf_BUILD_TESTS=OFF ^
  -Dprotobuf_BUILD_PROTOC_BINARIES=OFF ^
  -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
  -Ddlib_USE_CUDA=OFF ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_C_COMPILER=clang-cl ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_C_COMPILER_TARGET=arm64-pc-windows-msvc ^
  -DCMAKE_SYSTEM_NAME=Windows ^
  -DCMAKE_CXX_COMPILER_TARGET=arm64-pc-windows-msvc ^
  -DOCOS_ENABLE_CTEST=OFF -DBUILD_TESTING=OFF
```
