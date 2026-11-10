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
  -Wl,-rpath,"$PWD/cpp/macOS" -o ./systemone-test
./systemone-test
```

The optional graph argument runs real-model sanity, isolated-versus-multi-question,
usage, singleton/255-option, wrong-model and overlong-state checks. Use the same graph
layout described in [CLI.md](../../CLI.md#typesafe-compatible-endpoint). For example:

```sh
./systemone-test /models/d1/onnx/decoder_model_merged_quantized.onnx
```

HTTP/CLI regression checks use only Python's standard library, start their own
servers on available local ports and terminate them on exit:

```sh
python3 cpp/tests/systemone_http_test.py /path/to/rebuilt/onnx-genai \
  --kev /models/kev/onnx/model_q4.onnx \
  --d1 /models/d1/onnx/decoder_model_merged_quantized.onnx \
  --laya /models/laya/laya.onnx \
  --strands /models/strands-v1/onnx/model_q8.onnx
```

Without model arguments this checks unloaded-model and invalid-request handling,
the model catalogue, and existing endpoints' unloaded-model behavior.
The checked-in build and deployment libraries use ONNX Runtime 1.29.0, including
the IR-13 support needed by d1; a runtime environment override is not required.

For independent numeric parity, install `numpy`, `tokenizers` and `onnxruntime`
in a test environment and start a server with the desired model. The Python and
C++ runtime versions and intra-op thread counts must match: quantized kernels
can change across versions and threading configurations. Set `--threads` to the
server's printed `Detected ... Intra-Op threads` value (the script defaults to 2).
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

## Focused Laya checks (no model download)

The pure suite covers metadata validation, temperature/bucket boundaries and
fallback, Python-compatible structured rendering, MASK scrubbing, exact sequence
layout and strict no-truncation token limits. The optional generated fixture adds
real native tokenizer/ONNX inference, mixed-type batching and isolation, padding,
unpadded usage, calibration, wrong-backend aliases, nested special metadata,
malformed graph/metadata rejection and invalid numeric output rejection.

Compile using the same linked libraries as the existing focused suite:

```sh
clang++ -std=c++17 -O2 \
  -Icpp/include -Icpp/include/onnxruntime/core/session \
  cpp/tests/systemone_laya_test.cpp \
  -Lcpp/a -Lcpp/macOS -ljsoncpp -ltokenizers_cpp -ltokenizers_c \
  -lsentencepiece -lonnxruntime \
  -framework Security -framework CoreFoundation -framework SystemConfiguration \
  -Wl,-rpath,"$PWD/cpp/macOS" -o ./systemone-laya-test
./systemone-laya-test
```

In a test environment with `numpy`, `onnx`, `tokenizers` and `onnxruntime`, generate
the tiny fixtures in a fresh directory under the project and run native checks:

```sh
python3 cpp/tests/systemone_laya_fixture.py .laya-fixtures
./systemone-laya-test .laya-fixtures
python3 cpp/tests/systemone_http_test.py /path/to/rebuilt/onnx-genai --fixtures .laya-fixtures
```

The same generator creates tiny `dispatch-kev/` and `dispatch-d1/` fixtures.
The C++ suite exercises all three question types, input-dependent numeric
readouts, packed-versus-isolated results, usage, canonical model IDs and
cross-backend alias rejection for both existing adapters. This catches backend
enum-dispatch regressions without downloading Kev or d1 weights. The d1 fixture
uses an embedding graph returning `[1,L,1]`, a cache-free decoder accepting
`inputs_embeds`/`attention_mask`, and final-position logits `[1,1,V]`, as required
by the existing adapter. Kev emits `[1,L]` logits using configured delimiter
tokens. The HTTP `--fixtures` run covers unloaded handling and all three
backends' HTTP/CLI, concurrency, relative-path and invalid-request checks, while
deliberately skipping semantic model-quality assertions.

The graph has five dynamic inputs (`input_ids`, `attention_mask`, `marker_pos`,
`marker_mask`, `qtype`), float32 `logits[B,K]` and `act_probs[B,2]`. Its nonconstant
scores depend on token IDs, marker locations, qtypes and masked sequence sums.
The WordLevel tokenizer registers CLS/SEP/MASK/PAD with exact known IDs, and
includes automatic padding, truncation and postprocessing which the native
adapter must disable. Fixture metadata deliberately supplies different type
temperatures and bucket overrides.

For independent Python-tokenizer/ORT parity, start a rebuilt server with that
fixture (or an optional real Laya graph), then run:

```sh
/path/to/rebuilt/onnx-genai -s -p 8080 -D .laya-fixtures/model.onnx
python3 cpp/tests/systemone_laya_parity_test.py .laya-fixtures/model.onnx http://127.0.0.1:8080
```

The parity check independently builds each input, batches with two-dimensional
padding, applies calibration, and compares all three response types, unchanged
TypeSafe confidence, token counts, singleton behavior and mixed-versus-isolated
questions. It exercises structured Unicode and floating-point state/instructions
(including negative zero and Python fixed/scientific boundaries), ASCII DEL
escaping, injected MASK text, optional noul
labels, all option-count bucket boundaries and different sequence lengths.
No actual weights are required; real-weight parity is optional. The
`systemone_http_test.py --laya` and optional graph argument to
`systemone_test.cpp` are instead semantic sanity checks for actual weights.
They retain the Kev/d1 checks; Laya's 255-option request must be rejected when it
requires reference truncation. HTTP/CLI relative-path checks support both
root-level Laya graphs and existing `onnx/` graph layouts.

### Optional official PyTorch checkpoint comparison

This opt-in mode is separate from native/ORT parity. It additionally requires
`torch`, `transformers` and `safetensors`, and sufficient CPU memory for the full
checkpoint; these packages are not imported by the default checks. Supply an
already downloaded official directory containing:

```text
official-pytorch/
  rl_common.py
  rl_agent_config.json
  model.safetensors
  encoder/config.json
```

The ONNX bundle must supply its nested `tokenizer/` directory, including the
metadata needed by `AutoTokenizer`. The mode imports and executes the supplied
official `rl_common.py`, so use a trusted pinned copy. It makes no downloads and
requires no server:

```sh
python3 cpp/tests/systemone_laya_parity_test.py /models/laya/laya.onnx \
  --pytorch-dir /models/official-pytorch
```

It uses the official sequence builder, collator and model loader to compare a
four-row CPU batch (noul, two-choice, three-level score and twelve-choice) with a
structured Unicode state. It compares raw logits and action probabilities,
**not calibrated responses**, with default absolute maximum tolerances of
`1e-4` and `1e-5` respectively. The tolerances can be changed explicitly with
`--pytorch-logit-tolerance` and `--pytorch-act-tolerance`; this expensive mode
never runs unless `--pytorch-dir` is supplied.

One measured comparison used:

- ONNX: `receptron/laya-onnx`, revision
  `d42786bdff8a7550316339c5fbf8f46caa83ea1b`.
- Official checkpoint/source: `convaiinnovations/laya`, revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- PyTorch `2.15.0`, ONNX Runtime `1.29.0`, CPU.
- Maximum raw-logit difference: `2.574920654296875e-05`.
- Maximum action-probability difference: `0.0`.

These are observations for that four-row batch, not a universal `1e-5` bound,
exhaustive checkpoint equivalence, or a calibration-quality claim. In particular,
the official source's `choice:11+` temperature was `0.10058280825614929`,
whereas the community ONNX metadata used `0.945`. The native adapter and
native/ORT parity deliberately follow the supplied community metadata, so
calibrated twelve-choice probabilities need not match the official calibration
despite close raw logits. Structured choice/score descriptions use the native
API's JSON extension semantics; official Python `str`/`repr` rendering for such
descriptions is outside this measured ordinary-string-option comparison.

## Strands Decider v1

The native backend is the **current Qwen3.5 v1 fused option-scoring export**,
not the older hobson-v19/v21 checkpoint or a causal-LM letter readout.
The model-free C++ checks cover backend detection, ambiguity, temperatures and
fallback, metadata limits, exact prompt text, option order/spans, Unicode
whitespace, Python-compatible structured JSON rendering, and legacy Kev/d1
rendering/confidence/validation regressions.
Run `./systemone-test` without arguments after compiling it as shown above;
these checks require no model download.

Synthetic ONNX fixtures require test-only `numpy`, `tokenizers`, and `onnx`
packages. They exercise the real CLI/server without model downloads:

```sh
python3 cpp/tests/strands_fixture_test.py /path/to/rebuilt/onnx-genai
```

For real-model checks:

```sh
./systemone-test /models/strands-v1/onnx/model_q8.onnx
python3 cpp/tests/systemone_http_test.py /path/to/rebuilt/onnx-genai \
  --strands /models/strands-v1/onnx/model_q8.onnx
python3 cpp/tests/systemone_parity_test.py strands-decider \
  /models/strands-v1/onnx/model_q8.onnx http://127.0.0.1:8080 \
  --threads 8 --preparer ./systemone-test
python3 cpp/tests/systemone_parity_test.py strands-decider \
  /models/strands-v1/onnx/model_q8.onnx http://127.0.0.1:8080 \
  --threads 8 --timeout 600 --preparer ./systemone-test --request cpp/tests/strands_request.json
```

Replace `8` with the server's actual intra-op thread count. `--preparer` compares
**every token ID, answer position and option position**, not just the resulting
classifications, with the independent Python tokenizer's offset mapping. The
structured fixture includes NFC composition, non-ASCII text, emoji, added special
tokens, caller-supplied marker text, multiline option names, collapsed Unicode
whitespace, JSON floats and mixed question types/option counts. Production
inference uses separate state/question tokenization and one unpadded graph run
per question, so no question inherits another question's cache or option count.

### Provenance and coverage

Local macOS arm64 CPU evaluation uses bundled ONNX Runtime **1.29.0**, with
no runtime binary changes:

| Source | Revision |
|---|---|
| `opticsWolf/strands-decider-2B-qwen3.5-v1-2610-onnx` | `12143c0d1a5b1d0bc33178dfbabbdfec042ed0be` |
| Official `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` model card/config inspected | `216d1ccb46a9e9b036eb9187b248db98480f77d4` |
| Official `strands-labs/strands-decider` prompt, schema and strict-window source inspected | `13faf5214f7cbea07d005278f1aeaf3973611eba` |

Downloaded q8 SHA-256 hashes match the pinned export's manifest:

```text
model_q8.onnx       31fd51f3042a6bdc955b62cf30600fd1c9880c026e2eb554fa90914133795327
model_q8.onnx_data  11d611809d1d4f9a07bb059160c28dd678fe927bf4716a76f60ba2dd834df2a6
```

The export records its upstream adapter `source_revision` as `main`, not an
immutable commit; the exact adapter revision used by that conversion cannot be
established from this metadata. The official revision above identifies the
reference documentation inspected, not independently verified fused-weight
provenance.

Real q8 CPU checks cover all three primitives, mixed/isolated requests,
singleton and 255-choice inference, usage, alias/model selection, overlong
rejection, HTTP errors, concurrent requests, and CLI input/output/relative paths.
Independent Python ORT comparison uses the same pinned graph and runtime, with
a required maximum probability difference below `1e-4`; this establishes native
adapter parity, not fidelity of the export to original torch weights.
Both the basic three-question request and the five-question structured fixture
passed with exact token IDs/readout positions. The largest displayed probability
difference was `0.00000001` with matching eight-thread CPU configurations.

The official model card's rounded CPU example was also checked: q8 returned
`noul=0.848`, billing probability `0.764`, and score `1.033`, versus the documented
`0.858`, `0.773`, and `1.03`. This is an illustrative smoke check against rounded
published values, **not** a fresh upstream torch parity run. Existing TypeSafe
confidence formulas remain unchanged, including the score confidence formula;
probability/score agreement does not imply equality with another server's derived
confidence statistic.

Unverified here: fresh official torch-vs-ONNX parity, broader calibration/accuracy,
FP32/q4f16 execution, Windows, native 4D execution, and real-weight Kev/d1
regressions. Model-free legacy regression checks remain in the focused suite.
The community author's 11-case FP32/q8/q4f16 comparisons are reported separately
in [CLI.md](../../CLI.md#typesafe-compatible-endpoint), not claimed as local results.

## LiquidAI d1 (d1-2B, d1-Omni-2B)

The pure suite in `systemone_test.cpp` covers dispatch on `d1_attention`, config
and `calibration.json` validation, temperature buckets, special-token breaking,
option codes (both schemes), the noul/choice/score prompts, set packing
(`question_ids`, answer slots) and splitting rows at the token limit. With a
graph it adds the real-model checks; packed and isolated questions are compared
by decision only, since "set" attention lets the state read every packed question:

```sh
./systemone-test /models/d1-2B-ONNX/onnx/decoder_q8.onnx
python3 cpp/tests/systemone_http_test.py /path/to/rebuilt/onnx-genai \
  --lfm3 /models/d1-2B-ONNX/onnx/decoder_q8.onnx
```

`systemone_lfm3_parity_test.py` independently rebuilds the packed rows from the
checkpoint's published prompt and packing rules, runs the same decoder with
Python `onnxruntime` and requires equal input token counts and probability
differences below `1e-6` on four requests (mixed types, special tokens in caller
text, 40 options with structured descriptions, 21 questions over two rows).
Start the server with the graph and pass its printed intra-op thread count:

```sh
python3 cpp/tests/systemone_lfm3_parity_test.py \
  /models/d1-2B-ONNX/onnx/decoder_q8.onnx http://127.0.0.1:8080 --threads 8
```

On macOS arm64 with ONNX Runtime 1.29.0 and 8 threads, d1-2B q8 and
d1-Omni-2B q8 both matched to `3e-16`. This checks the C++ adapter against an independent reading of the
reference code on the same ONNX graph; it is not a comparison with LiquidAI's
PyTorch engine, whose code was not executed here.

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
  -Wl,-rpath,"$PWD/cpp/macOS" -o ./runtime-genai-test
./runtime-genai-test /models/tiny-random-gpt2-fp32
```

This loads the real decoder and generates three tokens from one input token,
checking the final sequence length and printing the loaded runtime version.
It is a compatibility smoke check, not a model-quality test.
