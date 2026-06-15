# _server
### Extends `_ONNX` to build and launch the `onnx-genai` command line.

> _server.new (controller : 4D.Class)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| controller | 4D.Class | -> | Optional custom controller class |

## Description

`_server` extends [`_ONNX`](_ONNX.md) (passing `"ONNX-genai"` as the executable name) and provides the `start` method, which assembles the full command string from an options object and launches it via `_CLI_Controller.execute`.

`_server` is never instantiated directly by application code. It is managed internally by the worker infrastructure; `cs.ONNX.ONNX` delegates to it via `cs.ONNX.workers.worker`.

### Methods

#### start (option : Object) → 4D.SystemWorker

Builds the CLI command and starts the server.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| option | Object | -> | Server options (see [ONNX](ONNX.md) for the full property list) |
| Result | 4D.SystemWorker | <- | The launched worker |

**Argument construction rules:**

The command always begins with `ONNX-genai -s` (server mode). Named model paths are appended using short flags. For embedding, reranking, and TTS models the flag points to the specific ONNX file inside the folder (`folder.file(model_name)`), not the folder itself:

| Option key | Flag | Argument |
| --- | --- | --- |
| `chat_completion_model` | `-m` | Folder path |
| `embeddings_model` | `-e` | `embeddings_model/embeddings_model_name` |
| `rerank_model` | `-r` | `rerank_model/rerank_model_name` |
| `tts_model` | `-T` | `tts_model/tts_model_name` |
| `port` | `-p` | Integer |
| `host` | `-h` | Text |
| `chat_template` | `-t` | Posted to stdin (not appended to command) |
| `pooling: "cls"` | `-c` | — |
| `pooling: "last-token"` | `-l` | — |
| `pooling: "multi-vector"` | `-b` | ColBERT |
| `pooling: "e2e"` | `-d` | Universal Sentence Encoder |
| `pooling: "mean"` | _(no flag)_ | Default |

After the named options, remaining keys in `option` are emitted as flags. Single-character keys use `-x value`; multi-character keys use `--key value`. Underscores in key names are replaced with hyphens. The following keys are reserved and excluded from the general loop: `i`, `o`, `s`, `j`, `c`, `l`, `b`, `d`, `_`, `h`, `host`, `p`, `port`, `t`, `chat_template`, `pooling`, `T`, `tts_model`, `tts_model_name`, `r`, `rerank_model`, `rerank_model_name`, `e`, `embeddings_model`, `embeddings_model_name`, `m`, `chat_completion_model`, `HF_TOKEN`.

If `option.chat_template` is set, it is posted to stdin rather than appended to the command string.

| Value type | CLI form |
| --- | --- |
| Real / Integer | `-x value` or `--key value` |
| Text | `-x escaped-value` or `--key escaped-value` |
| Boolean `True` | `-x` or `--key` (no value) |
| 4D.File (exists) | `-x escaped-path` or `--key escaped-path` |

## Examples

`_server` is used indirectly via `cs.ONNX.ONNX`:

```4d
var $onnx : cs.ONNX.ONNX
$onnx:=cs.ONNX.ONNX.new(8080; $huggingfaces; $homeFolder; $options; $event)
```

To terminate:

```4d
$onnx:=cs.ONNX.ONNX.new()
$onnx.terminate()
```

## See also

- [`_ONNX`](_ONNX.md) — parent class
- [`_CLI_Controller`](_CLI_Controller.md) — executes the assembled command
- [`_Model`](_Model.md) — calls `_server.start` after model download completes
