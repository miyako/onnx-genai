---
layout: default
---

![version](https://img.shields.io/badge/version-20%2B-E23089)
![platform](https://img.shields.io/static/v1?label=platform&message=mac-intel%20|%20mac-arm%20|%20win-64&color=blue)
[![license](https://img.shields.io/github/license/miyako/onnx-genai)](LICENSE)
![downloads](https://img.shields.io/github/downloads/miyako/onnx-genai/total)

# Use ONNX Runtime from 4D

#### Abstract

[**ONNX** (Open Neural Network eXchange)](https://github.com/onnx/onnx) is an open-source standard to represent machine learning models. It allows models trained in one framework (e.g. PyTorch) to be used in another framework (e.g. TensorFlow) with native hardware acceleration (NVIDIA, AMD, Intel, Apple Silicon, Qualcomm). 

#### Usage

Instantiate `cs.ONNX.ONNX` in your *On Startup* database method:

```4d
var $ONNX : cs.ONNX.ONNX

If (False)
    $ONNX:=cs.ONNX.ONNX.new()  //default
Else 
    var $homeFolder : 4D.Folder
    $homeFolder:=Folder(fk home folder).folder(".ONNX")
    var $file : 4D.File
    var $URL : Text
    var $port : Integer
    
    var $event : cs.event.event
    $event:=cs.event.event.new()
    /*
        Function onError($params : Object; $error : cs.event.error)
        Function onSuccess($params : Object; $models : cs.event.models)
        Function onData($request : 4D.HTTPRequest; $event : Object)
        Function onResponse($request : 4D.HTTPRequest; $event : Object)
        Function onTerminate($worker : 4D.SystemWorker; $params : Object)
    */
    
    $event.onError:=Formula(ALERT($2.message))
    $event.onSuccess:=Formula(ALERT($2.models.extract("name").join(",")+" loaded!"))
    $event.onData:=Formula(LOG EVENT(Into 4D debug message; This.file.fullName+":"+String((This.range.end/This.range.length)*100; "###.00%")))
    $event.onData:=Formula(MESSAGE(This.file.fullName+":"+String((This.range.end/This.range.length)*100; "###.00%")))
    $event.onResponse:=Formula(LOG EVENT(Into 4D debug message; This.file.fullName+":download complete"))
    $event.onResponse:=Formula(MESSAGE(This.file.fullName+":download complete"))
    $event.onTerminate:=Formula(LOG EVENT(Into 4D debug message; (["process"; $1.pid; "terminated!"].join(" "))))
    
    $port:=8080
    
    $folder:=$homeFolder.folder("microsoft/Phi-3.5-mini-instruct")
    $path:="cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
    $URL:="https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
    $chat:=cs.event.huggingface.new($folder; $URL; $path; "chat.completion")
    
    $folder:=$homeFolder.folder("all-MiniLM-L6-v2")
    $path:=""
    $URL:="ONNX-models/all-MiniLM-L6-v2-ONNX"
    $embeddings:=cs.event.huggingface.new($folder; $URL; $path; "embedding"; "model.onnx")
    
    $options:={}
    var $huggingfaces : cs.event.huggingfaces
    $huggingfaces:=cs.event.huggingfaces.new([$chat; $embeddings])
    
    $ONNX:=cs.ONNX.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
    
End if 
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
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1
  }'
```

```
curl -X POST http://127.0.0.1:8080/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"Rain won’t stop me. Wind won’t stop me. Neither will driving snow. Sweltering summer heat<2028>will only raise my determination. With a body built for endurance, a heart free of greed, I’ll never lose my temper, trying always to keep a quiet smile on my face."}'
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
var $onnx : cs.ONNX.ONNX
$onnx:=cs.ONNX.ONNX.new()
$onnx.terminate()
```

#### Chat Completions Model

Downloaded an ONNX model:

```
hf download microsoft/Phi-3.5-mini-instruct-onnx \
  --include "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*" \
  --local-dir .
```

Or,

```
hf download PleIAs/Baguettotron --local-dir ~/.onnx/baguettotron 
python -m onnxruntime_genai.models.builder -m ~/.onnx/baguettotron -o ~/.onnx/baguettotron-onnx -p int4 -e cpu
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

You need to place a `tokenizer.model` file for old Google models (T5, ALBERT) or a `tokenizer.json` file for model Hugging Face models (Qwen, GPT, BERT) next to the ONNX file.

The runtime will use this file to tokenise the input, run ONNX inference, mean pool, L2 noemalise the embeddings.

Alternatively you can use an "End-to-End" (E2E) model like [**universal-sentence-encoder-large-5-onnx**](https://huggingface.co/SamLowe/universal-sentence-encoder-large-5-onnx/) that takes raw string as input and returns vectors as output. In this scenario, pre-processing and post processing are both included in the ONNX inference.

> At its core, ONNX is a frameworks for maths, not text. An E2E model typically uses 
**`onnxruntime-extensions`** to handle string. However, the text processing is not as powerful as specialised tokenisers. It is normally better to use ONNX for the vector maths and handle string manipulation outside of ONNX.

Download and convert a model yourself with [optimum-cli](https://github.com/huggingface/optimum):


```sh
optimum-cli export onnx \
  --model sonoisa/sentence-bert-base-ja-mean-tokens-v2 \
  --task feature-extraction \
  --trust-remote-code \
  ~/.onnx/sonoisa/sentence-bert-base-ja-mean-tokens-v2-fp32

optimum-cli onnxruntime quantize \
  --onnx_model ~/.onnx/sonoisa/sentence-bert-base-ja-mean-tokens-v2-fp32 \
  --output ~/.onnx/sonoisa/sentence-bert-base-ja-mean-tokens-v2-onnx --avx2    
```

Or, 

Use [**`tf2onnx`**](https://github.com/onnx/tensorflow-onnx) to convert a **TensorFlow** to an ONNX E2E model:

```bash
pip install tf2onnx onnxruntime-extensions tensorflow-hub
python -m tf2onnx.convert \
    --saved-model /content/use_model \
    --output universal-sentence-encoder-large-5.onnx \
    --opset 12 \
    --extra_opset ai.onnx.contrib:1 \
    --tag serve
```

#### Models

||Model|Parameters|Size|Context&nbsp;Length|Vocabulary|Languages|
|-|-|-:|-:|-:|-:|:-:|
|🇺🇸|[Llama&nbsp;3.2&nbsp;1B](https://huggingface.co/keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu)|`1.2`|`1.87`|`128000`|`128256`|`8`
|🇺🇸|[Phi&nbsp;4&nbsp;Mini](https://huggingface.co/keisuke-miyako/Phi-4-mini-instruct-onnx-int4-cpu)|`1.2`|`4.93`|`128000`|`100352`|`24`|
|🇺🇸|[Phi&nbsp;3.5&nbsp;Mini](https://huggingface.co/keisuke-miyako/Phi-3.5-mini-instruct-onnx-int4-cpu)|`3.8`|`2.78`|`128000`|`32064`|`20`|
|🇺🇸|[Gemma&nbsp;2&nbsp;2B](https://huggingface.co/keisuke-miyako/gemma-2-2b-it-onnx-int4-cpu)|`2.6`|`4.04`|`8192`|`256128`|`English`|
|🇫🇷|[Lucie&nbsp;7B](https://huggingface.co/keisuke-miyako/Lucie-7B-Instruct-v1.1-onnx-int4-cpu)|`7.0`|`5.11`|`32000`|`65024`|`French`
|🇫🇷|[Baguettotron](https://huggingface.co/keisuke-miyako/Baguettotron-onnx-int4-cpu)|`0.3`|`1.79`|`8192`|`65000`|`European`
|🇨🇳|[Qwen&nbsp;3&nbsp;1.7B](https://huggingface.co/keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu)|`1.7`|`2.35`|`32768`|`151936`|`119`| 
|🇨🇳|[Qwen&nbsp;2.5&nbsp;1.5B](https://huggingface.co/keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu)|`1.5`|`1.94`|`128000`|`151936`|`29`| 
|🇺🇸|[Danube](https://huggingface.co/keisuke-miyako/h2o-danube-1.8b-chat-onnx-int4-cpu)|`1.8`|`1.43`|`16384`|`32000`|`English` 
|🇺🇸|[Danube&nbsp;2](https://huggingface.co/keisuke-miyako/h2o-danube2-1.8b-chat-onnx-int4-cpu)|`1.8`|`1.43`|`8192`|`32000`|`English`
|🇪🇺|[EuroLLM&nbsp;9B](https://huggingface.co/keisuke-miyako/EuroLLM-9B-Instruct-onnx-int4-cpu)|`9.1`|`7.51`|`4096`|`128000`|`European`|
|🇪🇺|[EuroLLM&nbsp;1.7B](https://huggingface.co/keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu)|`1.7`|`1.94`|`4096`|`128000`|`European`|
|🇺🇸|[SmolLM2&nbsp;1.7B](https://huggingface.co/keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu)|`1.7`|`1.48`|`8192`|`49152`|`6`
|🇺🇸|[Granite&nbsp;3.0&nbsp;2B](https://huggingface.co/keisuke-miyako/granite-3.0-2b-instruct-onnx-int4-cpu)|`2.5`|`1.99`|`4096`|`49155`|`12`
|🇯🇵|[ELYZA&nbsp;JP&nbsp;8B](https://huggingface.co/keisuke-miyako/Llama-3-ELYZA-JP-8B-onnx-int4-cpu)|`8.0`|`6.81`|`8192`|`128256`|`Japanese`&nbsp;`English`
|🇯🇵|[Swallow&nbsp;8B](https://huggingface.co/keisuke-miyako/Llama-3.1-Swallow-8B-Instruct-v0.3-onnx-int4-cpu)|`8.0`|`6.81`|`8192`|`128256`|`Japanese`&nbsp;`English`
|🇯🇵|[RakutenAI&nbsp;7B&nbsp;Chat](https://huggingface.co/keisuke-miyako/RakutenAI-7B-chat-onnx-int4-cpu)|`7.0`|`5.31`|`32768`|`48000`|`Japanese`&nbsp;`English`
|🇯🇵|[RakutenAI&nbsp;7B&nbsp;Instruct](https://huggingface.co/keisuke-miyako/RakutenAI-7B-instruct-onnx-int-cpu)|`7.0`|`5.31`|`32768`|`48000`|`Japanese`&nbsp;`English`
|🇯🇵|[Youko&nbsp;8B](https://huggingface.co/keisuke-miyako/llama-3-youko-8b-instruct-onnx-int4-cpu)|`8.0`|`6.81`|`8192`|`128256`|`Japanese`&nbsp;`English`
|🇯🇵|[Baku&nbsp;2B](https://huggingface.co/keisuke-miyako/gemma-2-baku-2b-it-onnx-int4-cpu)|`2.6`|`4.04`|`8192`|`256000`|`Japanese`&nbsp;`English`
|🇯🇵|[Youri&nbsp;7B&nbsp;Instruct](https://huggingface.co/keisuke-miyako/youri-7b-instruction-onnx-int4-cpu)|`7.0`|`4.66`|`4096`|`32000`|`Japanese`&nbsp;`English`
|🇯🇵|[Youri&nbsp;7Bnbsp;Chat](https://huggingface.co/keisuke-miyako/youri-7b-chat-onnx-int4-cpu)|`7.0`|`4.66`|`4096`|`32000`|`Japanese`&nbsp;`English`

#### Not Compatible

||Model|Parameters|Size|Context&nbsp;Length|Vocabulary|Languages|
|-|-|-:|-:|-:|-:|:-:|
|⚠️|[Ministral&nbsp;3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX)|`3.4`|`2.29`|`256000`|`131072`|`European`
|⚠️|[Gemma&nbsp;3&nbsp;1B](https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX)|`1.0`|`1.41`|`32768`|`262144`|`140`|
|⚠️|[Danube&nbsp;3](https://huggingface.co/keisuke-miyako/h2o-danube3-4b-chat-onnx-int4-cpu)|`4.0`|`2.9`|`8192`|`32000`|`English`
|⚠️|[Danube&nbsp;3.1](https://huggingface.co/keisuke-miyako/h2o-danube3.1-4b-chat-onnx-int4-cpu)|`4.0`|`2.9`|`8192`|`32000`|`English`
|⚠️|[CroissantLLMChat](https://huggingface.co/keisuke-miyako/CroissantLLMChat-v0.1-onnx-int4-cpu)|`1.3`|`1.07`|`2048`|`32000`|`French` 
|⚠️|[EXAONE&nbsp;3.5&nbsp;2.4B](https://huggingface.co/keisuke-miyako/EXAONE-3.5-2.4B-Instruct-onnx-int4-cpu)|`2.4`|`2.66`|`32768`|`102400`|`English`&nbsp;`Korean`
|⚠️|[InternLM&nbsp;1.8B](https://huggingface.co/keisuke-miyako/internlm2_5-1_8b-chat-onnx-in4-cpu)|`1.8`|`1.86`|`32768`|`92544`|`English`&nbsp;`Chinense`

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
