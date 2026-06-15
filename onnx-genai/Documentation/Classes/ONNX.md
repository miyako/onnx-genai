# ONNX
### The public entry point for managing an `onnx-genai` server instance.

> ONNX.new (port : Integer; huggingfaces : cs.event.huggingfaces; HOME : 4D.Folder; options : Object; event : cs.event.event)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| port | Integer | -> | Port to listen on (default: 8080) |
| huggingfaces | cs.event.huggingfaces | -> | Model download parameters |
| HOME | 4D.Folder | -> | Home folder (default: `Folder(fk home folder).folder(".ONNX")`) |
| options | Object | -> | Command-line options passed to `onnx-genai` |
| event | cs.event.event | -> | Callback functions (onError, onSuccess, onData, onResponse, onTerminate) |

## Description

`cs.ONNX.ONNX` is the main class you instantiate to manage an `onnx-genai` server process. It extends `_interface` and orchestrates the full lifecycle: checking whether a server is already running on the given port, downloading model files from Hugging Face if needed, and starting the server process in the background via a worker.

If a server is already running on the specified `port`, the constructor exits immediately without starting a new one.

Parameter defaults applied by the constructor:

- `port`: defaults to `8080` if `0`, negative, or greater than `65535`
- `HOME`: defaults to `Folder(fk home folder).folder(".ONNX")` if not provided or non-existent
- `huggingfaces`: defaults to `keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu` (chat) + `keisuke-miyako/embeddinggemma-300m-onnx` (embedding) if `Null` or empty; the default chat model also sets a Llama 3 chat template in `options.chat_template`

### options properties

| Property | Type | CLI flag | Description |
| --- | --- | --- | --- |
| chat_completion_model | 4D.Folder | `-m` | Path to an ONNX chat/generation model directory |
| embeddings_model | 4D.Folder | `-e` | Path to an ONNX embedding model directory |
| embeddings_model_name | Text | _(file within `-e`)_ | ONNX model filename inside `embeddings_model` (e.g. `"model.onnx"`) |
| rerank_model | 4D.Folder | `-r` | Path to an ONNX reranking model directory |
| rerank_model_name | Text | _(file within `-r`)_ | ONNX model filename inside `rerank_model` |
| tts_model | 4D.Folder | `-T` | Path to an ONNX text-to-speech model directory |
| tts_model_name | Text | _(file within `-T`)_ | ONNX model filename inside `tts_model` |
| port | Integer | `-p` | Port to listen on |
| host | Text | `-h` | Host address to bind |
| chat_template | Text | `-t` | Jinja2 chat template string; posted to stdin |
| pooling | Text | `-c` / `-l` / `-b` / `-d` | Pooling strategy (see table below) |
| HF_TOKEN | Text | _(auth header)_ | Hugging Face access token for gated models |

#### Pooling strategies

| pooling value | Flag | Notes |
| --- | --- | --- |
| `"cls"` | `-c` | CLS token |
| `"mean"` | _(no flag)_ | Default |
| `"last-token"` | `-l` | Last token |
| `"multi-vector"` | `-b` | ColBERT |
| `"e2e"` | `-d` | Universal Sentence Encoder |
| `"splade"` | _(not implemented)_ | — |

Additional options with underscore-separated names are passed through as long-form flags (`--flag value`) for single-character keys or (`--flag value`) for multi-character keys. Boolean `true` values produce flags without a value.

### Domain mapping

Each `cs.event.huggingface` descriptor has a `domain` property that determines which option keys the downloaded model folder and filename are assigned to:

| domain | options key (folder) | options key (filename) |
| --- | --- | --- |
| `"chat.completion"` | `chat_completion_model` | — |
| `"embedding"` | `embeddings_model` | `embeddings_model_name` |
| `"rerank"` | `rerank_model` | `rerank_model_name` |
| `"tts"` | `tts_model` | `tts_model_name` |

### API compatibility

| Endpoint | Availability |
| --- | --- |
| `/v1/chat/completions` | ✅ |
| `/v1/embeddings` | ✅ |
| `/v1/rerank` | ✅ |
| `/v1/audio/speech` | ✅ (TTS) |

## Examples

### Chat completion + embedding model

```4d
var $homeFolder : 4D.Folder
$homeFolder:=Folder(fk home folder).folder(".ONNX")

var $event : cs.event.event
$event:=cs.event.event.new()
$event.onError:=Formula(ALERT($2.message))
$event.onSuccess:=Formula(ALERT($2.models.extract("name").join(",")+" loaded!"))
$event.onData:=Formula(MESSAGE(This.file.fullName+":"+String((This.range.end/This.range.length)*100; "###.00%")))
$event.onResponse:=Formula(LOG EVENT(Into 4D debug message; This.file.fullName+":download complete"))
$event.onTerminate:=Formula(LOG EVENT(Into 4D debug message; (["process"; $1.pid; "terminated!"].join(" "))))

var $chat : cs.event.huggingface
$chat:=cs.event.huggingface.new(\
    $homeFolder.folder("microsoft/Phi-3.5-mini-instruct"); \
    "https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"; \
    "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"; \
    "chat.completion")

var $embeddings : cs.event.huggingface
$embeddings:=cs.event.huggingface.new(\
    $homeFolder.folder("all-MiniLM-L6-v2"); \
    "ONNX-models/all-MiniLM-L6-v2-ONNX"; \
    ""; \
    "embedding"; \
    "model.onnx")

var $huggingfaces : cs.event.huggingfaces
$huggingfaces:=cs.event.huggingfaces.new([$chat; $embeddings])

var $ONNX : cs.ONNX.ONNX
$ONNX:=cs.ONNX.ONNX.new(8080; $huggingfaces; $homeFolder; {}; $event)
```

### Test the chat completion endpoint

```
curl -X 'POST' \
  'http://127.0.0.1:8080/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1
  }'
```

### Test the embeddings endpoint

```
curl -X POST http://127.0.0.1:8080/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"The quick brown fox jumps over the lazy dog."}'
```

### Test the rerank endpoint

```
curl --request POST \
  --url http://127.0.0.1:8080/v1/rerank \
  --header 'Content-Type: application/json' \
  --data '{
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": [
      "Carson City is the capital city of the American state of Nevada.",
      "Washington, D.C. is the capital of the United States.",
      "Capital punishment has existed in the United States since before it was a country."
    ]
  }'
```

### Use with AI Kit

```4d
var $OpenAI : cs.AIKit.OpenAI
$OpenAI:=cs.AIKit.OpenAI.new({baseURL: "http://127.0.0.1:8080/v1"})

var $messages : Collection
$messages:=[]
$messages.push({role: "system"; content: "You are a helpful assistant."})
$messages.push({role: "user"; content: "Explain quantum computing in one sentence."})

var $params : cs.AIKit.OpenAIChatCompletionsParameters
$params:=cs.AIKit.OpenAIChatCompletionsParameters.new({model: ""})
$params.max_completion_tokens:=2048
$params.temperature:=0.7

var $result : cs.AIKit.OpenAIChatCompletionsResult
$result:=$OpenAI.chat.completions.create($messages; $params)
If ($result.success)
    ALERT($result.choice.message.text)
End if
```

### Terminate the server

```4d
var $onnx : cs.ONNX.ONNX
$onnx:=cs.ONNX.ONNX.new()
$onnx.terminate()
```

## See also

- [`_interface`](_interface.md) — parent class providing `terminate()` and TCP-check helpers
- [`_models`](_models.md) — download and model lifecycle management
- [`_Model`](_Model.md) — concrete model subclass
- [`_server`](_server.md) — CLI wrapper for `onnx-genai`
