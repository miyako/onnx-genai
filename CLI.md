# onnx-genai

ONNX Runtime GenAI Inference Engine

```
Usage:  onnx-genai -s -m chat_completion_model -e embedding_model -p port 

 -m path     : chat completion model
 -e path     : embedding model (pooling=mean)
 -r path     : rerank model
 -T path     : text to speach model
 -D path     : System One decision ONNX model (Kev, d1, English Laya or Strands Decider v1, text only)
 -t path     : chat template
 -j          : chat template from stdin
 -d          : pooling=e2e 
 -b          : pooling=multi-vector 
 -l          : pooling=last-token
 -c          : pooling=cls
 -s          : server
 -p          : server listening port (default=8080)
 -h host     : server host (default=127.0.0.1)
 -i          : input
 -o          : output (default=stdout)
 -           : use stdin for input
```

The CLI is built for `4` platforms:

- macOS Apple Silicon
- macOS Intel
- Windows AMD
- Windows ARM

## Dependencies

- `onnxruntime-genai-0.12.0`
- `onnxruntime-extensions-0.14.0`
- `onnxruntime-1.29.0` (CPU; macOS universal, Windows x64 and ARM64)

The bundled ONNX Runtime macOS library targets macOS 13.3 or later. Other
dependencies may impose higher deployment requirements. GenAI and Extensions
are separate libraries and retain the versions listed above.

## OpenAI Compatible Endpoints

- `/v1/models`
- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/audio/speech`
 
## Cohere Compatible Endpoints

- `/v1/rerank`

## MongoDB Compatible Endpoints

- `/v1/contextualizedembeddings`
- `/v1/contextualized/embeddings` (alias)

> [!TIP]
> The `input_type` property has no special effect. You must specifically add the prefixes "Represent the query for retrieving supporting documents: " and "Represent the document for retrieval: " if required by the model.
> [https://www.mongodb.com/docs/](https://www.mongodb.com/docs/api/doc/atlas-embedding-and-reranking-api/operation/operation-createcontextualizedembedding)

## TypeSafe Compatible Endpoint

- `POST /v1/systemone`

Load one decision model independently of chat, embeddings, reranking and speech:

```sh
onnx-genai -s -D /models/kev-0.6b-ONNX/onnx/model_q4.onnx
# Same adapter for kev-4b-ONNX:
onnx-genai -s -D /models/kev-4b-ONNX/onnx/model_q4.onnx
# d1 q8:
onnx-genai -s -D /models/d1-3B-ONNX/onnx/decoder_model_merged_quantized.onnx
# Laya English encoder:
onnx-genai -s -D /models/laya-onnx/laya.onnx
# Strands Decider v1 q8 (fused readout, not a causal-LM export):
onnx-genai -s -D /models/strands-decider-2B-qwen3.5-v1-2610-onnx/onnx/model_q8.onnx
```

Supported exports:
[Kev-0.6B](https://huggingface.co/onnx-community/kev-0.6b-ONNX),
[Kev-4B](https://huggingface.co/onnx-community/kev-4b-ONNX),
[d1-3B](https://huggingface.co/onnx-community/d1-3B-ONNX),
[Laya English](https://huggingface.co/receptron/laya-onnx/tree/d42786bdff8a7550316339c5fbf8f46caa83ea1b), and
[Strands Decider v1](https://huggingface.co/opticsWolf/strands-decider-2B-qwen3.5-v1-2610-onnx/tree/12143c0d1a5b1d0bc33178dfbabbdfec042ed0be).
These are decision models, not chat or embedding models; do not load them with
`-m`, `-e` or `-r`.

For Kev/d1, keep `config.json` and `tokenizer.json` in the model directory, with the selected
graph and **all its external data files** in `onnx/`. They may alternatively all
be placed in one directory. Kev, d1 and Strands are identified by exactly one
of `kev`, `d1` or `decider` metadata in `config.json`, respectively, not the
directory name or `model_type`.
Kev uses the selected graph directly. d1 additionally loads the matching
`embed_tokens_quantized.onnx` or `embed_tokens_q4.onnx` beside its selected decoder.
The text-only adapter does not load the vision encoder.
Strands uses the selected fused graph directly: `onnx/model_q8.onnx` needs
`onnx/model_q8.onnx_data` (~1.7 GB); `model.onnx` and `model_q4f16.onnx` likewise
need their corresponding external data. Keep the export's tokenizer assets and
configuration, not those from a differently named checkpoint. The official v1
checkpoint is a rank-48 LoRA plus dedicated option-scoring head on Qwen3.5-2B-Base;
the community export fuses both. Official adapter-only weights and older
hobson-v19/v21 exports are not interchangeable with this fused v1 layout.

The Laya adapter runs the 421M-parameter ModernBERT-large-based encoder plus
decision head on CPU, **English only**.
Use the community ONNX bundle `receptron/laya-onnx` pinned to revision
`d42786bdff8a7550316339c5fbf8f46caa83ea1b`; preserve its layout:

```text
laya-onnx/
  laya.onnx
  laya.onnx.data
  laya_config.json
  tokenizer/
    tokenizer.json
    tokenizer_config.json
```

`laya_config.json` alongside the selected graph identifies the native Laya
backend. Both tokenizer files are required; external `laya.onnx.data` must stay
beside the graph. Backend selection distinguishes `Kev`, `D1`, `Laya` and `Strands`;
ambiguous Kev/d1/Strands `config.json` metadata mixed with Laya metadata is rejected,
not resolved by preferring one backend. No backend flag is needed.
This supports the shipped English ONNX export, not the multilingual checkpoint.
An independent CPU comparison against official `convaiinnovations/laya` revision
`7b928d828b7b0e022f929d9bd2e44165aa270148` measured a maximum raw-logit difference
of `2.58e-5` in a four-row mixed-question batch using PyTorch 2.15.0 and
ONNX Runtime 1.29.0. This is a bounded smoke comparison, not universal `1e-5`
parity or model-quality validation. Published temperatures in `laya_config.json`
are consumed as-is and must be finite and positive. In particular,
`choice:11+ = 0.945` is the community export's calibration, not the official
checkpoint's `0.10058280825614929`; probabilities for those questions deliberately
follow the published ONNX bundle rather than the original checkpoint.

**Runtime requirement:** the published d1 export uses ONNX IR 13. The bundled
ONNX Runtime has been upgraded to **1.29.0** in both the C++ build dependencies
and the macOS/Windows deployment directories, so a runtime override is no longer
needed. Older installations using 1.23.2 cannot load this graph.
Laya and Strands use the same bundled CPU runtime **1.29.0**; neither requires a
runtime version bump.

For deployment, keep the 1.29.0 runtime alongside the rebuilt executable
(on Windows, deploy its corresponding provider DLLs too). System One still
requires an executable rebuilt from the updated source; replacing a runtime
library alone does not add the endpoint to an older executable.
Prefer Kev `q4`, Strands `q8` and d1
`q8` for initial CPU evaluation; other export precisions depend on runtime/kernel
support. Probabilities may differ across quantizations and runtime versions.
Quantized Kev probabilities can also vary slightly between a packed request and
the same questions evaluated separately, despite the graph's branch isolation.

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kev-latest",
    "state": {"message": "I was charged twice. Please refund the extra charge."},
    "questions": {
      "refund": {"type": "noul", "instructions": "Is a refund requested?"},
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments and refunds", "technical": "Software faults"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Routine", "Urgent", "Emergency"]
      }
    }
  }'
```

Use `kev-latest` for Kev, `d1-latest` for d1, `laya-latest` for Laya,
`strands-decider-latest` for Strands, or the model ID returned by `GET /v1/models`
(the model directory's basename). Family aliases only resolve to the loaded
backend (`kev`, `d1`, `laya` or `strands-decider`); an unknown model is rejected.
The response contains `model`, `answers`
keyed by question name, and `usage` with `input_tokens` and `output_tokens: 0`.
Noul returns `noul` (probability of true); Choice returns `choice`,
`probabilities` and `confidence`; Score returns a zero-based expected `score`,
`legend`, `probabilities` and `confidence`.
Laya uses this same response schema; no `act` fields are added.

Confidence follows TypeSafe's [documented formulas](https://docs.typesafe.ai/confidence):
Choice measures the largest probability above a uniform distribution, and Score
measures probability-weighted distance from the modal level. A single option/level
has confidence 1. These are distribution statistics, not guarantees of accuracy
or Jev-equivalent calibration. Strands divides each option logit by
`decider.temperature_by_kind[type]` (falling back to `decider.temperature`)
before softmax; missing, nonpositive or nonfinite effective temperatures fail
model loading. It does not generate answer letters or apply training-time
ordinal smoothing at inference.

The endpoint accepts string/object/array state and structured
instructions/descriptions. It rejects images, assets, `box`, unsupported fields,
duplicate JSON keys and inputs exceeding limits:

| Limit | Value |
|---|---|
| JSON body | 2 MiB |
| Named questions | 1–64 |
| Choice options | 1–255 |
| Score levels | Kev/d1/Laya: 1–10; Strands: 2–10 (upstream schema) |
| Kev state and each state-plus-question row | Model metadata, capped at 8,192 tokens |
| Kev complete packed request | 8,192 tokens |
| d1 each complete question prompt | Model context, capped at 8,192 tokens |
| d1 sum of question prompt tokens | 32,768 tokens |
| Laya complete question row (`max_len` in config) | 512 tokens |
| Laya head (`head_max_len` in config) | 192 tokens |
| Laya each option | 48 tokens |
| Laya padded question batch | 32,768 tokens |
| Strands each complete state-plus-question prompt | `decider.max_length` (v1: 4,096), capped at 8,192 tokens |
| Strands sum of complete prompt tokens | 32,768 tokens |

Over-limit inputs are rejected, **not truncated**. Kev shares the state within a
packed graph run. d1 and Strands evaluate each question independently, with fresh
caches (Strands' empty caches are inside the graph); their input usage counts
repeated state tokens. Mixed question kinds and option counts do not share
padding or logits. Strands follows the upstream engine's strict-window path:
the export's `max_question_fraction` is a truncation policy, not an additional
strict-mode limit, and is not applied. The exported models' long-context
behavior is not necessarily validated up to these serving limits.

Laya constructs each row in the reference order
`[CLS] type question: instructions [SEP] [MASK] options [SEP] state [SEP]`.
Structured state uses insertion-ordered JSON with Python-style spacing;
structured instructions additionally use ASCII JSON escapes, as in the Python
reference. Structured option descriptions use ordered JSON as an API extension
to the reference's text descriptions. Literal `[MASK]` in caller content is
replaced with a space so it cannot inject option markers.
It follows the reference equal-budget rule for options, but rejects the request
if any option would need shrinking (including to meet the 48-token option cap).
The remaining question/instruction head capacity is
`max(8, head_max_len - sum(option tokens including markers))`; a head that would
need truncation is rejected too. State plus the complete head and separators
must fit `max_len = 512`. These are **reject-not-truncate** limits, not permission
to silently shorten instructions, options or state.
Question rows are batched with right padding; the marker-position tensor has
width at least 2. The padded batch must fit 32,768 tokens, while
`usage.input_tokens` sums the **unpadded** row lengths (including repeated state).
The existing 1–64 question, 1–255 Choice option and 1–10 Score level limits
remain unchanged for Laya.

Invalid requests return HTTP `422` with a `detail` array, an unloaded decision
model returns `503`, and inference failures return `500`. Errors are logged.
Decision inference has its own mutex and does not repurpose the other model slots.
Without `-s`, `-D` selects decision evaluation from `-i`/stdin, with `-o` for a
JSON response file; failures return a nonzero exit status. If both `-m` and `-D`
are supplied in CLI mode, decision evaluation takes precedence.

Kev, the English Laya weights and Strands Decider (including the community fused
v1 export) are Apache-2.0 licensed. d1 uses the
[LFM Open License v1.0](https://huggingface.co/onnx-community/d1-3B-ONNX/blob/main/LICENSE),
including commercial-use restrictions; review it before deploying.

Strands calibration is fitted per primitive on held-out short classification,
not on arbitrary traffic or every quantization. The export author's limited
11-case CPU comparison reports maximum probability differences against the
upstream torch engine of 0.0021 (FP32, no argmax changes), 0.0504 (q8, no argmax
changes), and 0.2074 (q4f16, one argmax change). Classification agreement is not
numerical parity or evidence that confidence thresholds remain calibrated.
Local q8 native/Python ORT 1.29.0 parity passed the basic three-question and
structured five-question requests with exact token IDs/readout positions and a
largest displayed probability difference of `1e-8` on the pinned export above.
This is adapter parity, not a fresh official torch comparison.
**Windows and native 4D Strands execution remain unverified here.**
See [the focused checks](cpp/tests/README.md#strands-decider-v1) for pinned
revisions, local results and unverified paths.

## Converted ONNX Models

#### Quantisation

The `fp32` format is accurate but consumes `4` bytes per weight and pretty slow on a CPU. It is generally not suitable for production. You should only use it as a reference.

The `fp16` format consumes `2` bytes per weight. The CPU backend may be forced to perform calculations in `float32` except on a CPU like Apple Silicon that has native 16-bit maths which is not as fast as an NVIDIA GPU. It is usually best to avoid this format on a CPU.

The `int8` format takes advantage of `NEON` instructions on Apple Silicon and `AVX2` `AVX-512` `VNNI` instructions on Intel or AMD to **accelerate maths**. For encoders, the accuracy drop is said to be negligible ( less than `1%`). **You should always use the `int8` format on a PC or Mac with no GPU**. 

The `int4` format is designed to compress large language models. Just as a reference, a `7B` parameter in native `float32` format would requires `28GB` of memory, and on a CPU the data must go through the processor for every single token generation. `int4` reduces the bandwidth by `8`. The format internally groups multiple weights (e.g. `32`) to share a scale factor to maintain accuracy. However the quantisation is less precise compared to a GGUF (llama.cpp) model of a similar size. 

### Rerank

||`fp16`|`fp32`|`int8`|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`
|-|-:|-:|-:|-:|-:|-:
|[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)|[`45`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp16)|[`91`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp32)|[`23`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-int8)|`512`|`384`|`6`
|[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)|[`235`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp32)|[`118`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-int8)|`512`|`384`|`12`|
|[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)|[`1140`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp32)|[`569`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-int8)|`8192`|`1024`|`24`|
|[`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base)|[`556`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-int8)|`8192`|`768`|`12`|
|[`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large)|[`1120`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp32)|[`561`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-int8)|`8192`|`1024`|`24`
|[`jinaai/jina-reranker-v1-turbo-en`](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)|[`75`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp16)|[`151`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp32)|[`38`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-int8)|`8192`|`384`|`6`
|[`jinaai/jina-reranker-v3`](https://huggingface.co/jinaai/jina-reranker-v3)|||[`598`](https://huggingface.co/keisuke-miyako/jina-reranker-v3-onnx-int8)|`131072`|`1024`|`28`|
|[`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2)|[`299`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp16)|[`599`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-int8)|`8192`|`768`|`22`
|[`hotchpotch/japanese-reranker-tiny-v2`](https://huggingface.co/hotchpotch/japanese-reranker-tiny-v2)|[`58`](https://huggingface.co/keisuke-miyako/japanese-reranker-tiny-v2-onnx-fp16)|[`117`](https://huggingface.co/keisuke-miyako/japanese-reranker-tiny-v2-onnx-fp32)|[`29`](https://huggingface.co/keisuke-miyako/japanese-reranker-tiny-v2-onnx-int8)|`8192`|`256`|`3`
|[`hotchpotch/japanese-reranker-xsmall-v2`](https://huggingface.co/hotchpotch/japanese-reranker-xsmall-v2)|[`73`](https://huggingface.co/keisuke-miyako/japanese-reranker-xsmall-v2-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/japanese-reranker-xsmall-v2-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/japanese-reranker-xsmall-v2-onnx-int8)|`8192`|`256`|`10`
|[`hotchpotch/japanese-reranker-small-v2`](https://huggingface.co/hotchpotch/japanese-reranker-small-v2)|[`140`](https://huggingface.co/keisuke-miyako/japanese-reranker-small-v2-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/japanese-reranker-small-v2-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/japanese-reranker-small-v2-onnx-int8)|`8192`|`384`|`13`
|[`hotchpotch/japanese-reranker-base-v2`](https://huggingface.co/hotchpotch/japanese-reranker-base-v2)|[`265`](https://huggingface.co/keisuke-miyako/japanese-reranker-base-v2-onnx-fp16)|[`530`](https://huggingface.co/keisuke-miyako/japanese-reranker-base-v2-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/japanese-reranker-base-v2-onnx-int8)|`8192`|`512`|`19`
|[`cl-nagoya/ruri-v3-reranker-310m`](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)|[`630`](https://huggingface.co/keisuke-miyako/ruri-v3-reranker-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/ruri-v3-reranker-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/ruri-v3-reranker-310m-onnx-int8)|`8192`|`768`|`25`
|[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)|||[`598`](https://huggingface.co/keisuke-miyako/Qwen3-Reranker-0.6B-onnx-int8)|`32768`|`1024`|`28`|
|[`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B)|||[`4030`](https://huggingface.co/keisuke-miyako/Qwen3-Reranker-4B-onnx-int8)|`40960`|`2560`|`36`|
|[`zeroentropy/zerank-2`](https://huggingface.co/zeroentropy/zerank-2)|||[`4030`](https://huggingface.co/keisuke-miyako/zerank-2-onnx-int8)|`40960`|`2560`|`36`
|[`lightonai/ColBERT-Zero`](https://huggingface.co/lightonai/ColBERT-Zero)|[`298`](https://huggingface.co/keisuke-miyako/ColBERT-Zero-onnx-f16)|[`596`](https://huggingface.co/keisuke-miyako/ColBERT-Zero-onnx-f32)|[`151`](https://huggingface.co/keisuke-miyako/ColBERT-Zero-onnx-int8)|`8192`|`768`|`22`

### Embedding

||`fp16`|`fp32`|`int8`|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`|`pooling`
|-|-:|-:|-:|-:|-:|-:|-:
|[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small)|[`282`](https://huggingface.co/keisuke-miyako/mmBERT-small-onnx-f16)||[`141`](https://huggingface.co/keisuke-miyako/mmBERT-small-onnx-int8)|`8192`|`384`|`22`|`mean`
|[`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base)|[`616`](https://huggingface.co/keisuke-miyako/mmBERT-base-onnx-f16)||[`308`](https://huggingface.co/keisuke-miyako/mmBERT-base-onnx-int8)|`8192`|`768`|`22`|`mean`
|[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)|[`298`](https://huggingface.co/keisuke-miyako/ModernBERT-base-onnx-f16)|[`596`](https://huggingface.co/keisuke-miyako/ModernBERT-base-onnx-f32)|[`151`](https://huggingface.co/keisuke-miyako/ModernBERT-base-onnx-int8)|`8192`|`768`|`22`|`mean`
|[`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large)|[`789`](https://huggingface.co/keisuke-miyako/ModernBERT-large-onnx-f16)|[`1580`](https://huggingface.co/keisuke-miyako/ModernBERT-large-onnx-f32)|[`397`](https://huggingface.co/keisuke-miyako/ModernBERT-large-onnx-int8)|`8192`|`1024`|`28`|`mean`
|[`perplexity-ai/pplx-embed-v1-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)| | | [`614`](https://huggingface.co/keisuke-miyako/pplx-embed-v1-0.6b-onnx-int8) | `32768` | `1024` | `28` | `mean`
|[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)|[`66`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-int8)|`512`|`384`|`12`|`cls`
|[`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)|[`278`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp16)|[`435`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp32)|[`116`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-int8)|`512`|`768`|`12`|`cls`
|[`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)|[`668`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp32)|[`335`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-int8)|`512`|`1024`|`24`|`cls`
|[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)|[`1130`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp32)|[`568`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-int8)|`8192`|`1024`|`24`|`cls`
|[`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2)|[`66`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-int8)|`512`|`384`|`12`|`mean`
|[`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2)|[`218`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp16)|[`435`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp32)|[`109`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-int8)|`512`|`768`|`12`|`mean`
|[`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2)|[`668`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp32)|[`335`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-int8)|`512`|`1024`|`24`|`mean`
|[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)|[`235`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp32)|[`118`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-int8)|`512`|`384`|`12`|`mean`
|[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)|[`555`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-int8)|`512`|`768`|`12`|`mean`
|[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)|[`1120`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp32)|[`560`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-int8)|`512`|`1024`|`24`|`mean`
|[`nomic-ai/nomic-embed-text-v1`](https://huggingface.co/nomic-ai/nomic-embed-text-v1)|[`274`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-fp16)|[`547`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-fp32)|[`138`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-int8)|`8192`|`768`|`12`|`mean`
|[`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)|[`274`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-fp16)|[`547`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-fp32)|[`138`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-int8)|`8192`|`768`|`12`|`mean`
|[`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)|[`66`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-int8)|`512`|`384`|`12`|`cls`
|[`Snowflake/snowflake-arctic-embed-l`](https://huggingface.co/Snowflake/snowflake-arctic-embed-l)|[`668`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-fp32)|[`336`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-int8)|`512`|`1024`|`24`|`cls`
|[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)|[`45`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-fp16)|[`90`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-fp32)|[`22`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-int8)|`512`|`384`|`6`|`mean`
|[`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)|[`66`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-int8)|`512`|`384`|`12`|`mean`
|[`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)|[`607`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-fp16)|[`1210`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-fp32)|[`309`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-int8)|`2048`|`768`|`24`|`mean`
|[`retrieva-jp/amber-base`](https://huggingface.co/retrieva-jp/amber-base)|[`264`](https://huggingface.co/keisuke-miyako/amber-base-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/amber-base-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/amber-base-onnx-int8)|`8192`|`512`|`19`|`meqn`
|[`retrieva-jp/amber-large`](https://huggingface.co/retrieva-jp/amber-large)|[`629`](https://huggingface.co/keisuke-miyako/amber-large-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/amber-large-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/amber-large-onnx-int8)|`8192`|`768`|`25`|`mean`
|[`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)|[`278`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-fp16)|[`556`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-fp32)|[`146`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-int8)|`8192`|`768`|`12`|`cls`
|[`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)|[`873`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-fp16)|[`1750`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-fp32)|[`445`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-int8)|`8192`|`1024`|`24`|`cls`
|[`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)|[`628`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-fp32)|[`340`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-int8)|`8192`|`768`|`12`|`cls`
|[`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)|[`298`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-fp16)|[`596`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-int8)|`8192`|`768`|`22`|`cls`
|[`sbintuitions/modernbert-ja-30m`](https://huggingface.co/sbintuitions/modernbert-ja-30m)|[`73`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-int8)|`8192`|`256`|`10`|`mean`
|[`sbintuitions/modernbert-ja-70m`](https://huggingface.co/sbintuitions/modernbert-ja-70m)|[`140`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-int8)|`8192`|`384`|`13`|`mean`
|[`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m)|[`264`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-int8)|`8192`|`512`|`19`|`mean`
|[`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m)|[`629`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-int8)|`8192`|`768`|`25`|`mean`
|[`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m)|[`73`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-int8)|`8192`|`256`|`10`|`mean`
|[`cl-nagoya/ruri-v3-70m`](https://huggingface.co/cl-nagoya/ruri-v3-70m)|[`140`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-int8)|`8192`|`384`|`13`|`mean`
|[`cl-nagoya/ruri-v3-130m`](https://huggingface.co/cl-nagoya/ruri-v3-130m)|[`264`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-int8)|`8192`|`512`|`19`|`mean`
|[`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m)|[`629`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-int8)|`8192`|`768`|`25`|`mean`
|[`cl-nagoya/ruri-base-v2`](https://huggingface.co/cl-nagoya/ruri-base-v2)|[`221`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-fp16)|[`442`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-fp32)|[`111`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-int8)|`512`|`768`|`12`|`mean`
|[`cl-nagoya/ruri-large-v2`](https://huggingface.co/cl-nagoya/ruri-large-v2)|[`673`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-fp16)|[`1350`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-fp32)|[`338`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-int8)|`512`|`1024`|`24`|`mean`
|[`ibm-granite/granite-embedding-small-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)|[`95`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-fp16)|[`190`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-fp32)|[`48`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-int8)|`8192`|`384`|`12`|`cls`
|[`ibm-granite/granite-embedding-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-english-r2)|[`298`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-fp16)|[`596`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-int8)|`8192`|`768`|`22`|`cls`
|[`ibm-granite/granite-embedding-30m-english`](https://huggingface.co/ibm-granite/granite-embedding-30m-english)|[`60`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-fp16)|[`120`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-fp32)|[`30`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-int8)|`512`|`384`|`6`|`cls`
|[`ibm-granite/granite-embedding-125m-english`](https://huggingface.co/ibm-granite/granite-embedding-125m-english)|[`248`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-fp16)|[`496`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-fp32)|[`125`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-int8)|`512`|`768`|`12`|`cls`
|[`ibm-granite/granite-embedding-107m-multilingual`](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual)|[`213`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-fp16)|[`427`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-fp32)|[`107`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-int8)|`512`|`384`|`6`|`cls`
|[`ibm-granite/granite-embedding-278m-multilingual`](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)|[`555`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-int8)|`512`|`768`|`12`|`cls`
|[`Alibaba-NLP/gte-Qwen2-1.5B-instruct`](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)|||[`1680`](https://huggingface.co/keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx-int8)|`32768`|`1536`|`28`|`last-token`
|[`Alibaba-NLP/gte-Qwen2-7B-instruct`](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)|||[`7210`](https://huggingface.co/keisuke-miyako/gte-Qwen2-7B-instruct-onnx-int8)|`32768`|`3584`|`28`|`last-token`
|[`sbintuitions/sarashina-embedding-v1-1b`](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b)|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v1-1b-onnx-int8)|`8192`|`1792`|`24`|`last-token`
|[`sbintuitions/sarashina-embedding-v2-1b`](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b)|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v2-1b-onnx-int8)|`8192`|`1792`|`24`|`last-token`

### Chat Completion

||`int4`|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`
|-|-:|-:|-:|-:|
|[`MadeAgents/Hammer2.1-0.5b`](https://huggingface.co/MadeAgents/Hammer2.1-0.5b)|[`864`](https://huggingface.co/keisuke-miyako/Hammer2.1-0.5b-onnx-int4)|`32768`|`896`|`24`|
|[`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)|[`1860`](https://huggingface.co/keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4)|`8192`|`2048`|`16`|
|[`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)|[`3650`](https://huggingface.co/keisuke-miyako/Llama-3.2-3B-Instruct-onnx-int4)|`131072`|`3072`|`28`|
|[`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct)|[`4860`](https://huggingface.co/keisuke-miyako/Phi-4-mini-instruct-onnx-int4)|`131072`|`3072`|`32`|
|[`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)|[`2720`](https://huggingface.co/keisuke-miyako/Phi-3.5-mini-instruct-onnx-int4)|`131072`|`3072`|`32`|
|[`microsoft/Phi-4-mini-reasoning`](https://huggingface.co/microsoft/Phi-4-mini-reasoning)|[`4860`](https://huggingface.co/keisuke-miyako/Phi-4-mini-reasoning-onnx-int4)|`131072`|`3072`|`32`|
|[`microsoft/Phi-4-reasoning`](https://huggingface.co/microsoft/Phi-4-reasoning)|[`1090`](https://huggingface.co/keisuke-miyako/Phi-4-reasoning-onnx-int4)|`32768`|`5120`|`40`|
|[`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4)|[`1090`](https://huggingface.co/keisuke-miyako/Phi-4-onnx-int4)|`16384`|`5120`|`40`|
|[`microsoft/Phi-4-reasoning-plus`](https://huggingface.co/microsoft/Phi-4-reasoning-plus)|[`1090`](https://huggingface.co/keisuke-miyako/Phi-4-reasoning-plus-onnx-int4)|`32768`|`5120`|`40`|
|[`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)|[`5380`](https://huggingface.co/keisuke-miyako/gemma-3-4b-it-onnx-int4)|`131072`|`2560`|`34`|
|[`google/translategemma-4b-it`](https://huggingface.co/google/translategemma-4b-it)|[`5380`](https://huggingface.co/keisuke-miyako/translategemma-4b-it-onnx-int4)|`131072`|`2560`|`34`|
|[`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it)|[`1900`](https://huggingface.co/keisuke-miyako/gemma-3-1b-it-onnx-int4)|`32768`|`1152`|`26`|
|[`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it)|[`906`](https://huggingface.co/keisuke-miyako/gemma-3-270m-it-onnx-int4)|`32768`|`640`|`18`|
|[`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it)|[`906`](https://huggingface.co/keisuke-miyako/functiongemma-270m-it-onnx-int4)|`32768`|`640`|`18`|
|[`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)|[`4010`](https://huggingface.co/keisuke-miyako/gemma-2-2B-it-onnx-int4)|`8192`|`2304`|`26`|
|[`google/gemma-2-2b-jpn-it`](https://huggingface.co/google/gemma-2-2b-jpn-it)|[`4010`](https://huggingface.co/keisuke-miyako/gemma-2-2b-jpn-it-onnx-int4)|`8192`|`2304`|`26`|
|[`ibm-granite/granite-3.0-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct)|[`1990`](https://huggingface.co/keisuke-miyako/granite-3.0-2b-instruct-onnx-int4)|`4096`|`2048`|`40`|
|[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)|[`2020`](https://huggingface.co/keisuke-miyako/granite-3.3-2b-instruct-onnx-int4)|`131072`|`2048`|`40`|
|[`ibm-granite/granite-3.1-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)|[`5980`](https://huggingface.co/keisuke-miyako/granite-3.1-8b-instruct-onnx-int4)|`131072`|`4096`|`40`|
|[`ibm-granite/granite-3.2-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct)|[`5980`](https://huggingface.co/keisuke-miyako/granite-3.2-8b-instruct-onnx-int4)|`131072`|`4096`|`40`|
|[`ibm-granite/granite-3.3-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)|[`5980`](https://huggingface.co/keisuke-miyako/granite-3.3-8b-instruct-onnx-int4)|`131072`|`4096`|`40`|
|[`ibm-granite/granite-3.2-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct)|[`2020`](https://huggingface.co/keisuke-miyako/granite-3.2-2b-instruct-onnx-int4)|`131072`|`2048`|`40`|
|[`ibm-granite/granite-3.1-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct)|[`2020`](https://huggingface.co/keisuke-miyako/granite-3.1-2b-instruct-onnx-int4)|`131072`|`2048`|`40`|
|[`ibm-granite/granite-3.0-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct)|[`5920`](https://huggingface.co/keisuke-miyako/granite-3.0-8b-instruct-onnx-int4)|`4096`|`4096`|`40`|
|[`utter-project/EuroLLM-1.7B-Instruct`](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct)|[`1920`](https://huggingface.co/keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4)|`4096`|`2048`|`24`|
|[`utter-project/EuroLLM-9B-Instruct`](https://huggingface.co/utter-project/EuroLLM-9B-Instruct)|[`7490`](https://huggingface.co/keisuke-miyako/EuroLLM-9B-Instruct-onnx-int4)|`4096`|`4096`|`42`|
|[`h2oai/h2o-danube-1.8b-chat`](https://huggingface.co/h2oai/h2o-danube-1.8b-chat)|[`1430`](https://huggingface.co/keisuke-miyako/h2o-danube-1.8b-chat-onnx-int4)|`16384`|`2560`|`24`|
|[`h2oai/h2o-danube2-1.8b-chat`](https://huggingface.co/h2oai/h2o-danube2-1.8b-chat)|[`1430`](https://huggingface.co/keisuke-miyako/h2o-danube2-1.8b-chat-onnx-int4)|`8192`|`2560`|`24`|
|[`PleIAs/Pleias-RAG-1B`](https://huggingface.co/PleIAs/Pleias-RAG-1B)|[`1290`](https://huggingface.co/keisuke-miyako/Pleias-RAG-1B-onnx-int4)|`4096`|`2048`|`22`|
|[`PleIAs/Baguettotron`](https://huggingface.co/PleIAs/Baguettotron)|[`353`](https://huggingface.co/keisuke-miyako/Baguettotron-onnx-int4)|`4096`|`576`|`80`|
|[`deepseek-ai/deepseek-coder-1.3b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)|[`1080`](https://huggingface.co/keisuke-miyako/deepseek-coder-1.3b-instruct-onnx-int4)|`16384`|`2048`|`24`|
|[`OpenLLM-France/Lucie-7B-Instruct-v1.1`](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1)|[`5110`](https://huggingface.co/keisuke-miyako/Lucie-7B-Instruct-v1.1-onnx-int4)|`32000`|`4096`|`32`|
|[`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)|[`1470`](https://huggingface.co/keisuke-miyako/SmolLM2-1.7B-onnx-int4)|`8192`|`2048`|`24`|
|[`01-ai/Yi-Coder-1.5B-Chat`](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat)|[`1440`](https://huggingface.co/keisuke-miyako/Yi-Coder-1.5B-Chat-onnx-int4)|`131072`|`2048`|`24`|
|[`OpenLLM-France/Claire-7B-FR-Instruct-0.1`](https://huggingface.co/OpenLLM-France/Claire-7B-FR-Instruct-0.1)|[`6920`](https://huggingface.co/keisuke-miyako/Claire-7B-FR-Instruct-0.1-onnx-int4)|`2048`|`4544`|`32`|
|[`tiiuae/Falcon3-7B-Instruct`](https://huggingface.co/tiiuae/Falcon3-7B-Instruct)|[`7460`](https://huggingface.co/keisuke-miyako/Falcon3-7B-Instruct-onnx-int4)|`32768`|`3072`|`28`|
|[`tiiuae/Falcon3-3B-Instruct`](https://huggingface.co/tiiuae/Falcon3-3B-Instruct)|[`3230`](https://huggingface.co/keisuke-miyako/Falcon3-3B-Instruct-onnx-int4)|`32768`|`3072`|`22`|
|[`tiiuae/Falcon3-1B-Instruct`](https://huggingface.co/tiiuae/Falcon3-1B-Instruct)|[`1670`](https://huggingface.co/keisuke-miyako/Falcon3-1B-Instruct-onnx-int4)|`8192`|`2048`|`18`|
|[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)|[`4210`](https://huggingface.co/keisuke-miyako/Qwen3-4B-Instruct-2507-onnx-int4)|`262144`|`2560`|`36`|
|[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)|[`1920`](https://huggingface.co/keisuke-miyako/Qwen2.5-1.5B-Instruct-onnx-int4)|`32768`|`1536`|`28`|
|[`Qwen/Qwen2.5-7B-Instruct-1M`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)|[`7120`](https://huggingface.co/keisuke-miyako/Qwen2.5-7B-Instruct-1M-onnx-int4)|`1010000`|`3584`|`28`|
|[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)|[`3190`](https://huggingface.co/keisuke-miyako/Qwen2.5-3B-Instruct-onnx-int4)|`32768`|`2048`|`36`|
|[`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B)|[`1920`](https://huggingface.co/keisuke-miyako/Qwen2.5-1.5B-onnx-int4)|`131072`|`1536`|`28`|
|[`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B)|[`3190`](https://huggingface.co/keisuke-miyako/Qwen2.5-3B-onnx-int4)|`32768`|`2048`|`36`|
|[`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B)|[`6670`](https://huggingface.co/keisuke-miyako/Qwen2.5-7B-onnx-int4)|`131072`|`3584`|`28`|
|[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)|[`2340`](https://huggingface.co/keisuke-miyako/Qwen3-1.7B-onnx-int4)|`40960`|`2048`|`28`|
|[`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)|[`4210`](https://huggingface.co/keisuke-miyako/Qwen3-4B-Thinking-2507-onnx-int4)|`262144`|`2560`|`36`|
|[`elyza/Llama-3-ELYZA-JP-8B`](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B)|[`6800`](https://huggingface.co/keisuke-miyako/Llama-3-ELYZA-JP-8B-onnx-int4)|`8192`|`4096`|`32`|
|[`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3`](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3)|[`6800`](https://huggingface.co/keisuke-miyako/Llama-3.1-Swallow-8B-Instruct-v0.3-onnx-int4)|`8192`|`4096`|`32`|
|[`Rakuten/RakutenAI-7B-chat`](https://huggingface.co/Rakuten/RakutenAI-7B-chat)|[`5290`](https://huggingface.co/keisuke-miyako/RakutenAI-7B-chat-onnx-int4)|`32768`|`4096`|`32`|
|[`Rakuten/RakutenAI-7B-instruct`](https://huggingface.co/Rakuten/RakutenAI-7B-instruct)|[`5290`](https://huggingface.co/keisuke-miyako/RakutenAI-7B-instruct-onnx-int4)|`32768`|`4096`|`32`|
|[`rinna/llama-3-youko-8b-instruct`](https://huggingface.co/rinna/llama-3-youko-8b-instruct)|[`6800`](https://huggingface.co/keisuke-miyako/llama-3-youko-8b-instruct-onnx-int4)|`8192`|`4096`|`32`|
|[`rinna/gemma-2-baku-2b-it`](https://huggingface.co/rinna/gemma-2-baku-2b-it)|[`4010`](https://huggingface.co/keisuke-miyako/gemma-2-baku-2b-it-onnx-int4)|`8192`|`2304`|`26`|
|[`rinna/youri-7b-instruction`](https://huggingface.co/rinna/youri-7b-instruction)|[`4660`](https://huggingface.co/keisuke-miyako/youri-7b-instruction-onnx-int4)|`4096`|`4096`|`32`|
|[`rinna/youri-7b-chat`](https://huggingface.co/rinna/youri-7b-chat)|[`4660`](https://huggingface.co/keisuke-miyako/youri-7b-chat-onnx-int4)|`4096`|`4096`|`32`|
|[`cyberagent/calm2-7b-chat`](https://huggingface.co/cyberagent/calm2-7b-chat)|[`5300`](https://huggingface.co/keisuke-miyako/calm2-7b-chat-onnx-int4)|`32768`|`4096`|`32`|
|[`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)|[`1970`](https://huggingface.co/keisuke-miyako/DeepSeek-R1-Distill-Qwen-1.5B-onnx-int4)|`131072`|`1536`|`28`|
|[`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)|[`6670`](https://huggingface.co/keisuke-miyako/DeepSeek-R1-Distill-Qwen-7B-onnx-int4)|`131072`|`3584`|`28`|
|[`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)|[`7290`](https://huggingface.co/keisuke-miyako/DeepSeek-R1-0528-Qwen3-8B-onnx-int4)|`131072`|`4096`|`36`|
