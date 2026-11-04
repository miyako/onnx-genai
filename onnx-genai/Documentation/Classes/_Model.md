# _Model
### Extends `_models` to assign model folder paths and filenames by domain, then launch `onnx-genai`.

> _Model.new (port : Integer; huggingfaces : cs.event.huggingfaces; options : Object; formula : 4D.Function; event : cs.event.event)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| port | Integer | -> | Port to listen on |
| huggingfaces | cs.event.huggingfaces | -> | Model download parameters |
| options | Object | -> | Command-line options (mutated; model folder and filename keys are set as downloads complete) |
| formula | 4D.Function | -> | Internal response callback |
| event | cs.event.event | -> | Callback functions, or `Null` for no callbacks |

## Description

`_Model` is the concrete implementation of [`_models`](_models.md). After calling `Super` it immediately triggers `download()`, unless `offline` is `true` and there is nothing to download (in which case the `start` method launches the server).

It overrides these methods from `_models`:

### models () → cs.event.models

Returns a `cs.event.models` collection built from the internal `_models` list (repository identifiers of the form `user/repo`). Each entry is wrapped as a `cs.event.model` with `isHuggingFace: True`.

### onDownload (oid : Text; downloadId : Text)

Overrides the virtual base. When a file completes downloading, resolves the parent folder of that file and assigns it — along with the model filename — to the appropriate `options` keys based on the `domain` of the descriptor. First-write wins (keys are only set if `Null`):

| domain | options key (folder) | options key (filename) |
| --- | --- | --- |
| `"chat.completion"` | `chat_completion_model` | — |
| `"embedding"` | `embeddings_model` | `embeddings_model_name` |
| `"rerank"` | `rerank_model` | `rerank_model_name` |
| `"systemone"` | `decision_model` | `decision_model_name` |
| `"tts"` | `tts_model` | `tts_model_name` |

For `systemone`, the folder is the descriptor's repository-root `4D.Folder`, not the parent of whichever file finishes first. The selected filename is preserved, including a relative `onnx/` prefix, so metadata or sibling-graph download order cannot change the model path.

This same mapping handles the fused Strands Decider v1 export: `opticsWolf/strands-decider-2B-qwen3.5-v1-2610-onnx`, domain `systemone`, selected name `onnx/model_q8.onnx`. Root configuration and tokenizer downloads do not overwrite that selection. Startup waits for the entire download queue, including external weights, to finish. Backend selection is handled by the executable using `config.json`'s `decider` metadata; no Strands-specific 4D option is added.

The completed record is looked up by `downloadId` when given (by `oid` otherwise), so files that share an OID map to their own domain. Then calls `Super.onDownload($oid; $downloadId)` to remove the entry from `files` and trigger `start()` when the queue empties.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| oid | Text | -> | OID of the completed download |
| downloadId | Text | -> | Destination identity assigned by `download()` |

### _onNothingToDownload ()

Called by `download()` when no file needs downloading. Calls `start()` when at least one of `chat_completion_model`, `embeddings_model`, `rerank_model`, `decision_model` or `tts_model` in `options` is an existing `4D.Folder`. Otherwise it calls `event.onError` with code `2` (including the given model paths, if any) and does not launch. Offline startup is handled by the `start` method and is not affected.

### start ()

Overrides the virtual base. Does nothing while `files` still has pending downloads (the last completion starts the server). Launches at most once per instance; later calls are ignored. Creates a `cs.ONNX.workers.worker` wrapping `_server`, calls `worker.start(port, options)`, then fires `event.onSuccess` with the current options and model list. `onSuccess` means the launch was requested, not that the server is ready; poll `/v1/models`.

### Properties

In addition to properties inherited from `_models`:

| Property | Type | Description |
| --- | --- | --- |
| chat_completion_model | 4D.Folder | Set after the first `chat.completion`-domain download completes |
| embeddings_model | 4D.Folder | Set after the first `embedding`-domain download completes |
| embeddings_model_name | Text | ONNX filename inside `embeddings_model` |
| rerank_model | 4D.Folder | Set after the first `rerank`-domain download completes |
| rerank_model_name | Text | ONNX filename inside `rerank_model` |
| decision_model | 4D.Folder | Repository root from the first `systemone`-domain download |
| decision_model_name | Text | Selected ONNX path relative to `decision_model` |
| tts_model | 4D.Folder | Set after the first `tts`-domain download completes |
| tts_model_name | Text | ONNX filename inside `tts_model` |

## See also

- [`_models`](_models.md) — parent class
- [`_server`](_server.md) — launched by `start()`; uses `*_model_name` to construct file paths for `-e`, `-r`, `-D`, and `-T` flags
- [`ONNX`](ONNX.md) — public entry point
