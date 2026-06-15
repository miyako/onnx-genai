# _Model
### Extends `_models` to assign model folder paths and filenames by domain, then launch `onnx-genai`.

> _Model.new (port : Integer; huggingfaces : cs.event.huggingfaces; options : Object; formula : 4D.Function; event : cs.event.event)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| port | Integer | -> | Port to listen on |
| huggingfaces | cs.event.huggingfaces | -> | Model download parameters |
| options | Object | -> | Command-line options (mutated; model folder and filename keys are set as downloads complete) |
| formula | 4D.Function | -> | Internal response callback |
| event | cs.event.event | -> | Callback functions |

## Description

`_Model` is the concrete implementation of [`_models`](_models.md). After calling `Super` it immediately triggers `download()` unless `offline` is `true`.

It overrides three methods from `_models`:

### models () → cs.event.models

Returns a `cs.event.models` collection built from the internal `_models` list (repository identifiers of the form `user/repo`). Each entry is wrapped as a `cs.event.model` with `isHuggingFace: True`.

### onDownload (oid : Text)

Overrides the virtual base. When a file completes downloading, resolves the parent folder of that file and assigns it — along with the model filename — to the appropriate `options` keys based on the `domain` of the descriptor. First-write wins (keys are only set if `Null`):

| domain | options key (folder) | options key (filename) |
| --- | --- | --- |
| `"chat.completion"` | `chat_completion_model` | — |
| `"embedding"` | `embeddings_model` | `embeddings_model_name` |
| `"rerank"` | `rerank_model` | `rerank_model_name` |
| `"tts"` | `tts_model` | `tts_model_name` |

Then calls `Super.onDownload($oid)` to remove the entry from `files` and trigger `start()` when the queue empties.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| oid | Text | -> | OID of the completed download |

### start ()

Overrides the virtual base. Creates a `cs.ONNX.workers.worker` wrapping `_server`, calls `worker.start(port, options)`, then fires `event.onSuccess` with the current options and model list.

### Properties

In addition to properties inherited from `_models`:

| Property | Type | Description |
| --- | --- | --- |
| chat_completion_model | 4D.Folder | Set after the first `chat.completion`-domain download completes |
| embeddings_model | 4D.Folder | Set after the first `embedding`-domain download completes |
| embeddings_model_name | Text | ONNX filename inside `embeddings_model` |
| rerank_model | 4D.Folder | Set after the first `rerank`-domain download completes |
| rerank_model_name | Text | ONNX filename inside `rerank_model` |
| tts_model | 4D.Folder | Set after the first `tts`-domain download completes |
| tts_model_name | Text | ONNX filename inside `tts_model` |

## See also

- [`_models`](_models.md) — parent class
- [`_server`](_server.md) — launched by `start()`; uses `*_model_name` to construct file paths for `-e`, `-r`, and `-T` flags
- [`ONNX`](ONNX.md) — public entry point
