---
layout: default
---

![version](https://img.shields.io/badge/version-20%2B-E23089)
![platform](https://img.shields.io/static/v1?label=platform&message=mac-intel%20|%20mac-arm%20|%20win-64&color=blue)
[![license](https://img.shields.io/github/license/miyako/onnx-genai)](LICENSE)
![downloads](https://img.shields.io/github/downloads/miyako/onnx-genai/total)

# Use ONNX Runtime from 4D

#### Abstract

[**ONNX** (Open Neural Network eXchange)](https://github.com/onnx/onnx) is an open-source standard to represent machine learning models. It allows models trained in one framework (e.g. PyTorch) to be used in another framework (e.g. TensorFlow) with native hardware accelaration (NVIDIA, AMD, Intel, Apple Silicon, Qualcomm).

The ONNX Runtime does not need to be updated in order to understand and support new model architectures. The architecture is effectively encoded in the ONNX graph itself (you do need to design an exporter for each new model architecture). 

#### Usage

Instantiate `cs.onnx.onnx` in your *On Startup* database method:

```4d
//T.B.D.
```

Unless the server is already running (in which case the costructor does nothing), the following procedure runs in the background:

1. The specified model is downloaded via HTTP
2. The `onnx-genai` program is started

Now you can test the server:

```
curl -X 'POST' \
  'http://127.0.0.1:8080/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain quantum computing in one sentence."
      }
    ],
    "temperature": 0.7
  }'
```

Or, use AI Kit:

```4d
var $ChatCompletionsParameters : cs.AIKit.OpenAIChatCompletionsParameters
$ChatCompletionsParameters:=cs.AIKit.OpenAIChatCompletionsParameters.new({model: ""})

$ChatCompletionsParameters.max_completion_tokens:=2048
$ChatCompletionsParameters.n:=1
$ChatCompletionsParameters.temperature:=0.7
//%W-550.26
$ChatCompletionsParameters.top_k:=50
$ChatCompletionsParameters.top_p:=0.9
//%W+550.26
$ChatCompletionsParameters.body:=Formula($0:={\
top_k: This.top_k; \
top_p: This.top_p; \
temperature: This.temperature; \
n: This.n; \
max_completion_tokens: This.max_completion_tokens})
$messages:=[]
$messages.push({role: "system"; content: "You are a helpful assistant."})
$messages.push({role: "user"; content: "The window was shattered. Inside the room were 3 cats, a piano, 1 million dollars, a baseball bat, a bar of soap. What happened?"})

var $OpenAI : cs.AIKit.OpenAI
$OpenAI:=cs.AIKit.OpenAI.new({baseURL: "http://127.0.0.1:8080/v1"})

var $ChatCompletionsResult : cs.AIKit.OpenAIChatCompletionsResult
$ChatCompletionsResult:=$OpenAI.chat.completions.create($messages; $ChatCompletionsParameters)
If ($ChatCompletionsResult.success)
    ALERT($ChatCompletionsResult.choice.message.text)
End if 
```

Finally to terminate the server:

```4d
var $onnx : cs.onnx
$onnx:=cs.onnx.new()
$onnx.terminate()
```

#### Chat Completions Model

As mentioned earlier, the model needs to be converted to ONNX. 

Downloaded a converted ONNX model:

```
hf download microsoft/Phi-3.5-mini-instruct-onnx \
  --include "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*" \
  --local-dir .
```

Or, convert one yourself with [optimum-cli](https://github.com/huggingface/optimum):


```sh
optimum-cli export onnx --model BAAI/bge-base-en-v1.5 onnx_output_dir/
```

The ONNX format is based on **Protocol Buffers**, which has a hard limit of `2GB` for a single file. ONNX splits the model into a `.onnx` file (the graph) and a `.data` file (the weights).

To download a specific file, Hugging Face uses the /resolve/ endpoint. The URL structure is:

```
https://huggingface.co/[REPO_ID]/resolve/[BRANCH]/[FILE_PATH]
```

Example:

```
TOKEN="your_hf_token_here"
curl -H "Authorization: Bearer $TOKEN" -L -O [URL]
curl -L -O https://huggingface.co/SamLowe/universal-sentence-encoder-large-5-onnx/resolve/main/model.onnx
curl -s "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" | \
jq -r '.[] | select(.type=="file") | .path' | \
while read -r file; do
    echo "Downloading $file..."
    curl -L --create-dirs -o "$file" "https://huggingface.co/$REPO/resolve/main/$file"
```

#### Embeddings Model

Alternatively you can use an "End-to-End" (E2E) model like [**universal-sentence-encoder-large-5-onnx**](https://huggingface.co/SamLowe/universal-sentence-encoder-large-5-onnx/) that takes raw string as input and returns vectors as output. 

> At its core, ONNX is a frameworks for maths, not text. An E2E model typically uses 
`**onnxruntime-extensions**` to handle string. However, the text processing is not as powerful as specialised tokenisers. It is normally better to use ONNx for the vector maths and handle string externally.

Use [`**tf2onnx**`](https://github.com/onnx/tensorflow-onnx) to convert a **TensorFlow** to an ONNX E2E model:

```bash
pip install tf2onnx onnxruntime-extensions tensorflow-hub
python -m tf2onnx.convert \
    --saved-model /content/use_model \
    --output universal-sentence-encoder-large-5.onnx \
    --opset 12 \
    --extra_opset ai.onnx.contrib:1 \
    --tag serve
```

#### AI Kit compatibility

The API is compatibile with [Open AI](https://platform.openai.com/docs/api-reference/embeddings). 

|Class|API|Availability|
|-|-|:-:|
|Models|`/v1/models`|✅|
|Chat|`/v1/chat/completions`|✅|
|Images|`/v1/images/generations`||
|Moderations|`/v1/moderations`||
|Embeddings|`/v1/embeddings`|✅|
|Files|`/v1/files`||
