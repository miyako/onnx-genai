---
layout: default
---

![version](https://img.shields.io/badge/version-20%2B-E23089)
![platform](https://img.shields.io/static/v1?label=platform&message=mac-intel%20|%20mac-arm%20|%20win-64&color=blue)
[![license](https://img.shields.io/github/license/miyako/onnx-genai)](LICENSE)
![downloads](https://img.shields.io/github/downloads/miyako/onnx-genai/total)

# Use ONNX Runtime from 4D

#### Abstract


#### Usage

Instantiate `cs.onnx.onnx` in your *On Startup* database method:

```4d

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

#### Models

Use [optimum-cli](https://github.com/huggingface/optimum) to convert a specific model to ONNX:

```sh
optimum-cli export onnx --model BAAI/bge-base-en-v1.5 onnx_output_dir/
```

Finally to terminate the server:

```4d
var $onnx : cs.onnx
$onnx:=cs.onnx.new()
$onnx.terminate()
```

#### AI Kit compatibility

The API is compatibile with [Open AI](https://platform.openai.com/docs/api-reference/embeddings). 

|Class|API|Availability|
|-|-|:-:|
|Models|`/v1/models`|✅|
|Chat|`/v1/chat/completions`|✅|
|Images|`/v1/images/generations`||
|Moderations|`/v1/moderations`||
|Embeddings|`/v1/embeddings`||
|Files|`/v1/files`||
