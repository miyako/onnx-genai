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

#### ONNX chat completion performance on Apple Silicon

You will likely notice the CPU usage jump to `400%` on M1. This means all `4` performance cores are running at full speed. This is because the ONNX Runtime generic `cpu` provider is not fully optimised for the ARM64 NEON instruction set used by Apple Silicon. The CPU can't perform the math fast enough for the `GeLU` activation functions and `RoPE` rotary embeddings. 

That means the CPU must de-quantise the small blocks of `Int4` data to `Float32`, do the matrix multiplication math, and re-quantise or move to the next block. That would amount to about `4` to `8` tokens per second on M1.

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
     -d '{"input":"Rain won’t stop me. Wind won’t stop me. Neither will driving snow. Sweltering summer heat will only raise my determination. With a body built for endurance, a heart free of greed, I’ll never lose my temper, trying always to keep a quiet smile on my face."}'
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

To test the `/rerank` endpoint:

```
curl --request POST \
  --url http://127.0.0.1:8080/v1/rerank \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": [
      "Carson City is the capital city of the American state of Nevada.",
      "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
      "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
      "Capital punishment (the death penalty) has existed in the United States since before the United States was a country."
    ]
  }'
```

Finally to terminate the server:

```4d
var $onnx : cs.ONNX.ONNX
$onnx:=cs.ONNX.ONNX.new()
$onnx.terminate()
```

#### Chat Completions Model

Download and convert a model with [optimum](https://github.com/huggingface/optimum):

```py
from google.colab import drive
drive.mount('/content/drive')
```

```py
!pip install "optimum[onnxruntime-gpu]" transformers
!pip install numpy onnxruntime-genai
!pip install onnx_ir
```

```py
from google.colab import userdata
from huggingface_hub import login
# Retrieve the token from the secrets manager
hf_token = userdata.get('HF_TOKEN')

# Log in to Hugging Face
login(token=hf_token)

print("Logged in successfully!")
```

```py
from huggingface_hub import snapshot_download
import os
import json

# --- STEP 1: Download the model locally ---
print("Downloading model snapshot to local disk...")
# We use Python here so we don't need the flag in the command line later
local_model_path = snapshot_download(
    repo_id="cyberagent/calm2-7b-chat",
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Skip useless files to save speed
)
print(f"Model downloaded to: {local_model_path}")

# --- STEP 2: Run the Builder on the local files ---
# We pass the local path variable to the command using '$'
print("Starting INT4 Conversion...")

!python -m onnxruntime_genai.models.builder \
    -m "$local_model_path" \
    -o "/content/drive/My Drive/calm2_INT4_CPU" \
    -p int4 \
    -e cpu --extra_options trust_remote_code=true
```

#### Embeddings Model

Download and convert a model with [optimum](https://github.com/huggingface/optimum):


```py
from google.colab import drive
drive.mount('/content/drive')
```

```py
!pip install "optimum[onnxruntime-gpu]" transformers
!pip install numpy onnxruntime-genai
!pip install onnx_ir  
```

```py
!optimum-cli export onnx \
  --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
  --task feature-extraction \
  --trust-remote-code \
  "/content/drive/My Drive/Alibaba-NLP/gte-Qwen2-1.5B-instruct-fp32"

!HF_TRUST_REMOTE_CODE=1

!optimum-cli onnxruntime quantize \
  --onnx_model "/content/drive/My Drive/Alibaba-NLP/gte-Qwen2-1.5B-instruct-fp32" \
  --output "/content/drive/My Drive/Alibaba-NLP/gte-Qwen2-1.5B-instruct-onnx" --avx2
  ```

You need to place a `tokenizer.model` file for old Google models (T5, ALBERT) or a `tokenizer.json` file for Hugging Face models (Qwen, GPT, BERT) next to the ONNX file.

The runtime will use this file to tokenise the input, run ONNX inference, apply pooling, and L2 noemalise the embeddings.

> At its core, ONNX is a frameworks for maths, not text. An E2E model typically uses 
**`onnxruntime-extensions`** to handle string. However, the text processing is not as powerful as specialised tokenisers. It is normally better to use ONNX for the vector maths and handle string manipulation outside of ONNX.

Alternatively, convert a **TensorFlow** to an ONNX E2E model with [**`tf2onnx`**](https://github.com/onnx/tensorflow-onnx):

```py
from google.colab import drive
drive.mount('/content/drive')
```

```py
!pip install tf2onnx onnxruntime-extensions kagglehub
```

```py
import tensorflow as tf
import tensorflow_hub as hub
import os
import subprocess

# 1. Configuration
TF_MODEL_URL = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2"
SAVED_MODEL_DIR = "/content/use_model"
OUTPUT_ONNX_PATH = "/content/drive/universal-sentence-encoder-large-5.onnx"

# Ensure output directory exists (if using Google Drive, make sure it's mounted)
output_dir = os.path.dirname(OUTPUT_ONNX_PATH)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print(f"1. Downloading and saving model to {SAVED_MODEL_DIR}...")

# 2. Load and Save the model
# Loading the Hub module and saving it ensures we have a standard SavedModel structure
module = hub.load(TF_MODEL_URL)
tf.saved_model.save(module, SAVED_MODEL_DIR)

print(f"2. Converting model to ONNX at {OUTPUT_ONNX_PATH}...")

# 3. Convert using tf2onnx
# We use subprocess.run to pass the python variables safely to the shell command
command = [
    "python", "-m", "tf2onnx.convert",
    "--saved-model", SAVED_MODEL_DIR,
    "--output", OUTPUT_ONNX_PATH,
    "--opset", "13",                  # Opset 13 is recommended for text/string support
    "--tag", "serve"
]

result = subprocess.run(command, capture_output=True, text=True)

# 4. Check results
if result.returncode == 0:
    print(result.stderr)  # tf2onnx logs often go to stderr
    print(f"\nSuccess! Model saved to: {OUTPUT_ONNX_PATH}")
else:
    print("Error during conversion:")
    print(result.stderr)
    print(result.stdout)
```

An "End-to-End" (E2E) model that takes raw string as input and returns vectors as output. In this scenario, pre-processing, inference, and post processing are all baked into the model.

#### Chat Completion Models

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
|🇯🇵|[Youri&nbsp;7B&nbsp;Chat](https://huggingface.co/keisuke-miyako/youri-7b-chat-onnx-int4-cpu)|`7.0`|`4.66`|`4096`|`32000`|`Japanese`&nbsp;`English`
|🇯🇵|[Calm2&nbsp;7B&nbsp;Chat](https://huggingface.co/keisuke-miyako/calm2-7b-chat-onnx)|`7.0`|`5.3`|`32768`|`65024`|`Japanese`&nbsp;`English`

#### Not Compatible

||Model|Parameters|Size|Context&nbsp;Length|Vocabulary|Languages|
|-|-|-:|-:|-:|-:|:-:|
|⚠️|[Ministral&nbsp;3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX)|`3.4`|`2.29`|`256000`|`131072`|`European`
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
