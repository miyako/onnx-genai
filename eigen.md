---
layout: default
---

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

