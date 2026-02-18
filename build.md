---
layout: default
---

# Memo: Build Instructions

### `onnxruntime-extensions`

## macOS

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
```

```
cmake --build . --config Release --parallel
```

### `onnxruntime-genai`

## macOS (ARM)

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
    --build_dir build \
    --config Release \
    --osx_arch arm64 # --osx_arch x86_64
```

## macOS (Intel)

```
python3 build.py \
    --build_dir build \
    --config Release \
    --osx_arch x86_64
```

There is no osx-x64 prebuilt runtime in `~/Desktop/onnxruntime-genai/build/Release/_deps/ortlib-src/runtimes/osx-x64/native` so you need to supply that manually.


Force `protobuf` rebuild by hiding it.

```
brew unlink protobuf
deactivate
```

```
export CMAKE_POLICY_VERSION_MINIMUM=3.5
```

```
./build.sh \
    --config Release \
    --osx_arch x86_64 \
    --build_shared_lib \
    --parallel \
    --skip_tests \
    --compile_no_warning_as_error \
    --skip_submodule_sync
```
