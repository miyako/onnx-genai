# System One checks

Build the server with the existing Xcode or Visual Studio project. On macOS,
compile the focused C++ tests from the repository root:

```sh
clang++ -std=c++17 -O2 \
  -Icpp/include -Icpp/include/onnxruntime/core/session \
  cpp/tests/systemone_test.cpp \
  -Lcpp/a -Lcpp/macOS -ljsoncpp -ltokenizers_cpp -ltokenizers_c \
  -lsentencepiece -lonnxruntime \
  -framework Security -framework CoreFoundation -framework SystemConfiguration \
  -Wl,-rpath,"$PWD/cpp/macOS" -o /tmp/systemone-test
/tmp/systemone-test
```

The optional graph argument runs real-model sanity, isolated-versus-multi-question,
usage, singleton/255-option, wrong-model and overlong-state checks. Use the same graph
layout described in [CLI.md](../../CLI.md#typesafe-compatible-endpoint). For example:

```sh
/tmp/systemone-test /models/d1/onnx/decoder_model_merged_quantized.onnx
```

HTTP/CLI regression checks use only Python's standard library, start their own
servers on available local ports and terminate them on exit:

```sh
python3 cpp/tests/systemone_http_test.py /path/to/rebuilt/onnx-genai \
  --kev /models/kev/onnx/model_q4.onnx \
  --d1 /models/d1/onnx/decoder_model_merged_quantized.onnx
```

Without model arguments this checks unloaded-model and invalid-request handling,
the model catalogue, and existing endpoints' unloaded-model behavior.
The checked-in build and deployment libraries use ONNX Runtime 1.29.0, including
the IR-13 support needed by d1; a runtime environment override is not required.

For independent numeric parity, install `numpy`, `tokenizers` and `onnxruntime`
in a test environment and start a server with the desired model. The Python and
C++ runtime versions must match: quantized kernels can change across versions.
This check independently constructs the three text prompts/readouts, verifies
input token counts, and requires probability differences below `1e-4`:

```sh
python3 cpp/tests/systemone_parity_test.py kev /models/kev/onnx/model_q4.onnx http://127.0.0.1:8080
python3 cpp/tests/systemone_parity_test.py d1 /models/d1/onnx/decoder_model_merged_quantized.onnx http://127.0.0.1:8081
```

The Python parity packages are test-only dependencies, not server dependencies.
The separate packed-versus-isolated question checks allow `1e-3` for Kev and
`1e-4` for d1. Kev's exported `accuracy_level=4` quantized kernels can produce
slightly shape-dependent probabilities; on the Kev-4B fixture the maximum
difference was approximately `0.00065`, while independent same-shape ONNX parity
was below `1e-4`.
Kev-4B uses the same tests and adapter as Kev-0.6B but requires its own graph and
external weights; testing one checkpoint does not establish the other's quality.

## GenAI runtime compatibility

The runtime upgrade also needs to preserve the separate GenAI library's
generation path. Download the files from the upstream
[`tiny-random-gpt2-fp32` fixture](https://github.com/microsoft/onnxruntime-genai/tree/f3a57ba8a904477f6c4ae786af9055bdeb65a621/test/test_models/hf-internal-testing/tiny-random-gpt2-fp32)
into a local directory, then compile and run:

```sh
clang++ -std=c++17 -O2 \
  -Icpp/include -Icpp/include/onnxruntime/core/session \
  cpp/tests/runtime_genai_test.cpp \
  -Lcpp/macOS -lonnxruntime-genai -lonnxruntime \
  -Wl,-rpath,"$PWD/cpp/macOS" -o /tmp/runtime-genai-test
/tmp/runtime-genai-test /models/tiny-random-gpt2-fp32
```

This loads the real decoder and generates three tokens from one input token,
checking the final sequence length and printing the loaded runtime version.
It is a compatibility smoke check, not a model-quality test.
