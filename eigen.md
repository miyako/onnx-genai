---
layout: default
---

# Eigen

```
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
arch -x86_64 /usr/local/bin/brew install gcc

```

## Intel

```
cmake -S . -B build_amd \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_C_FLAGS="-O3 -march=haswell" \
  -DCMAKE_CXX_FLAGS="-O3 -march=haswell" \
  -DCMAKE_Fortran_COMPILER=/usr/local/bin/gfortran
cmake --build build_amd --config Release  
```

## Apple Silicon

```
cmake -S . -B build_arm 
  -DCMAKE_BUILD_TYPE=Release 
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DACCELERATE_FRAMEWORK=ON
  -DBLAS=Accelerate
  -DCMAKE_C_FLAGS="-O3 -mcpu=apple-m1" 
  -DCMAKE_CXX_FLAGS="-O3 -mcpu=apple-m1"
cmake --build build_arm --config Release  
```  

## Xcode

* Other C Flags: `-Xpreprocessor` `-fopenmp`
* Optimization Level: `-O3`

# Tokenizers

```
cmake -S . -B build_arm  
    -DCMAKE_OSX_ARCHITECTURES=arm64 
    -DCMAKE_BUILD_TYPE=Release 
    -DMLC_ENABLE_SENTENCEPIECE_TOKENIZER=ON  
    -DCMAKE_CXX_FLAGS=-O3  
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build_arm --config Release
```

```
export CARGO_BUILD_TARGET=x86_64-apple-darwin
cmake -S . -B build_amd  
    -DCMAKE_OSX_ARCHITECTURES=x86_64 
    -DCMAKE_BUILD_TYPE=Release 
    -DMLC_ENABLE_SENTENCEPIECE_TOKENIZER=ON  
    -DCMAKE_C_FLAGS="-O3 -march=haswell" 
    -DCMAKE_CXX_FLAGS="-O3 -march=haswell" 
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DTOKENIZERS_CPP_CARGO_TARGET=x86_64-apple-darwin
cmake --build build_amd --config Release
```

## Visual Studio

```powershell
$Env:CARGO_BUILD_TARGET = "x86_64-pc-windows-msvc"
# 1. Tell Cargo to build for x64, not the native ARM64
$Env:CARGO_BUILD_TARGET = "x86_64-pc-windows-msvc"
# 2. Run CMake with the x64 Architecture flag
cmake -S . -B build  `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_CXX_FLAGS="/bigobj /openmp /O2 /fp:fast /arch:AVX2 /Ob2" `
    -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_POLICY_VERSION_MINIMUM="3.5" 
cmake --build build --config Release
```
