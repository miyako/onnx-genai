# ONNX Runtime GenAI
Local inference engine

**acknowledgements**: [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)

#### Abstract

[**ONNX** (Open Neural Network eXchange)](https://github.com/onnx/onnx) is an open-source standard to represent machine learning models. It allows models trained in one framework (e.g. PyTorch) to be used in another framework (e.g. TensorFlow) with native hardware acceleration (NVIDIA, AMD, Intel, Apple Silicon, Qualcomm). 

> The inference engine used in this component is configured to primarily run on CPU cores, for maximum compatibility. 

ONNX Runtime **1.29.0** is bundled for macOS universal and Windows x64/ARM64.
The runtime's macOS deployment target is 13.3; other dependencies may require a
newer macOS version. See [build instructions](build.md) for runtime provenance
and replacement workflows.

#### Quantisation

The `fp32` format is accurate but consumes `4` bytes per weight and pretty slow on a CPU. It is generally not suitable for production. You should only use it as a reference.

The `fp16` format consumes `2` bytes per weight. The CPU backend may be forced to perform calculations in `float32` except on a CPU like Apple Silicon that has native 16-bit maths which is not as fast as an NVIDIA GPU. It is usually best to avoid this format on a CPU.

The `int8` format takes advantage of `NEON` instructions on Apple Silicon and `AVX2` `AVX-512` `VNNI` instructions on Intel or AMD to **accelerate maths**. For encoders, the accuracy drop is said to be negligible ( less than `1%`). **You should always use the `int8` format on a PC or Mac with no GPU**. 

The `int4` format is designed to compress large language models. Just as a reference, a `7B` parameter in native `float32` format would requires `28GB` of memory, and on a CPU the data must go through the processor for every single token generation. `int4` reduces the bandwidth by `8`. The format internally groups multiple weights (e.g. `32`) to share a scale factor to maintain accuracy. However the quantisation is less precise compared to a GGUF (llama.cpp) model of a similar size. 

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

#### AI Kit compatibility

The API is compatible with the following [Open AI](https://platform.openai.com/docs/api-reference/) endpoints: 

|Class|API|Availability|
|-|-|:-:|
|Models|`/v1/models`|✅|
|Chat|`/v1/chat/completions`|✅|
|Images|`/v1/images/generations`||
|Moderations|`/v1/moderations`||
|Embeddings|`/v1/embeddings`|✅|
|Files|`/v1/files`||

In addition, the following endpoints are available:

|API|Alias|
|-|-|
|`/v1/audio/speech`|
|`/v1/rerank`||
|`/v1/contextualizedembeddings`|`/v1/contextualized/embeddings`
|`/v1/systemone`|Text-only TypeSafe decision API (Kev, d1, English Laya or Strands Decider v1)|

Load a decision model independently with `-D path/to/model.onnx`. Kev-0.6B,
Kev-4B, d1-3B, English Laya and the fused Strands Decider 2B Qwen3.5 v1 ONNX
exports support `noul`, `choice` and `score` questions.
Use `kev-latest`, `d1-latest`, `laya-latest` or `strands-decider-latest` for
the corresponding loaded backend (`kev`, `d1`, `laya` or `strands-decider`).
For Strands, `q8` is recommended for initial CPU evaluation.
See [CLI: TypeSafe Compatible Endpoint](CLI.md#typesafe-compatible-endpoint) for
model layouts, request examples, limits and runtime requirements. The bundled
CPU runtime is **1.29.0**, including support for d1's ONNX IR 13; Laya and
Strands require no runtime bump. Rebuild the CLI/server to enable new backends.
For 4D downloading/loading, use the `systemone` domain and a relative
graph name as shown in [the ONNX class documentation](onnx-genai/Documentation/Classes/ONNX.md).

Laya uses the **English-only, 421M ModernBERT-large** community export
[`receptron/laya-onnx`](https://huggingface.co/receptron/laya-onnx/tree/d42786bdff8a7550316339c5fbf8f46caa83ea1b),
pinned to `d42786bdff8a7550316339c5fbf8f46caa83ea1b`. Load
`-D /models/laya-onnx/laya.onnx` on CPU and use `laya-latest` or the model
directory's basename. Keep `laya_config.json` and `laya.onnx.data` beside the
graph and preserve `tokenizer/tokenizer.json` and
`tokenizer/tokenizer_config.json`. Mixed Kev/d1/Strands and Laya metadata is rejected.
Laya preserves the TypeSafe response/confidence contract without adding `act`
fields. Inputs exceeding its configured 512-token rows, 192-token head,
48-token options or 32,768-token padded batch are rejected, not truncated;
see the CLI documentation for the equal-budget head rules.
The export's published temperatures are used unchanged (finite positive values
required), including community-calibrated `choice:11+ = 0.945`, which differs
from the official checkpoint. An independent mixed-question CPU comparison
measured a maximum raw-logit difference of `2.58e-5` against the pinned official
PyTorch checkpoint; this does not establish universal parity, model quality or
multilingual compatibility.

Strands uses the pinned
[`opticsWolf` fused v1 export](https://huggingface.co/opticsWolf/strands-decider-2B-qwen3.5-v1-2610-onnx/tree/12143c0d1a5b1d0bc33178dfbabbdfec042ed0be).
Calibration is per primitive and not validated for arbitrary traffic or every
quantization. Local q8 native/Python ORT parity passed the basic three-question
and structured five-question requests with exact token IDs/readout positions
and a largest displayed probability difference of `1e-8` on ORT 1.29.0.
This is adapter parity, not a fresh official torch comparison or accuracy claim;
Windows and native 4D Strands execution remain unverified here.
See [focused checks and provenance](cpp/tests/README.md#strands-decider-v1).
