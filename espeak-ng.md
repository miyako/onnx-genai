---
layout: default
---

# espeak-ng

```
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=../install -DUSE_SPEECHPLAYER=FALSE -DUSE_LIBSONIC=FALSE -DUSE_PCAUDIO=FALSE 
cmake --build . --config Release
cmake --install . --config Release
```
