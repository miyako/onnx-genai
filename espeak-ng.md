---
layout: default
---

# espeak-ng

```
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --config Release
cmake --install . --config Release
```

-DWITH_PCAUDIOLIB=OFF
-framework AudioUnit -framework CoreAudio
winmm.lib
