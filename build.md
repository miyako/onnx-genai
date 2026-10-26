---
layout: default
---

# Memo: Build Instructions

## Automated ONNX Runtime libraries

Once the workflow is on the repository's default branch, run
**Actions > ONNX Runtime libraries > Run workflow** and enter an official
stable version without the `v` prefix. The default is **1.29.0**, which has been
verified with the d1 adapter; this does not automatically upgrade the runtime
bundled with the application.

The workflow `.github/workflows/onnxruntime.yml` produces:

| Artifact | Origin | Contents |
| --- | --- | --- |
| `onnxruntime-<version>-macos-universal` | Source build, pinned to the release commit | Universal arm64 + x86_64 `lib/libonnxruntime.dylib`, C/C++ headers, licenses |
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

The artifact helpers use only Python's standard library (Python 3.11+).
Run their focused checks with:

```sh
python3 -m unittest discover -s tools/runtime -p '*_test.py'
actionlint .github/workflows/onnxruntime.yml
```

This is a manual, read-only workflow: it uploads artifacts retained for 30 days,
does not commit binaries, does not publish a release, and does not run the 4D
component release workflow. macOS libraries are **ad-hoc signed**, not Developer
ID signed or notarized. Native Windows ARM64 runner availability depends on the
repository's GitHub Actions plan.

To adopt an artifact, first exercise the models you intend to use. Copy its
runtime library files from `lib/` to the appropriate application's runtime
directory (and the Windows `.lib` files to the corresponding C++ link-library
directory if rebuilding). Keep Windows DLLs from the same package together.
For a macOS source build, copy `libonnxruntime.dylib` to `cpp/macOS/`; packaged
applications use `onnx-genai/Resources/bin/macOS/`. Windows application runtime
directories are `onnx-genai/Resources/bin/Windows/` and `WindowsARM/`.

This workflow builds/packages **ONNX Runtime itself**, not `onnx-genai.exe`, the
macOS CLI, ONNX Runtime GenAI, Extensions, or the other static dependencies.
Their existing versions are intentionally left unchanged. A runtime update
must not be taken as proof that all those consumers remain compatible.

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
