# _models
### Abstract base class that resolves Hugging Face model metadata and manages file downloads.

> _models.new (port : Integer; huggingfaces : cs.event.huggingfaces; options : Object; formula : 4D.Function; event : cs.event.event)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| port | Integer | -> | Port stored in `options.port` |
| huggingfaces | cs.event.huggingfaces | -> | One or more Hugging Face model descriptors |
| options | Object | -> | Server options object (mutated to add `port`; when `event` is not `Null`, also `onTerminate`, `onStdErr`, `onStdOut`) |
| formula | 4D.Function | -> | Internal response callback forwarded to the download helper |
| event | cs.event.event | -> | Callback functions, or `Null` for no callbacks |

## Description

`_models` is the download-and-startup orchestrator. On construction it iterates the `huggingfaces` collection, queries the Hugging Face API to resolve file and directory metadata, and builds an internal `files` queue. If the API cannot be reached for an entry (`request.response.status = Null`) the class sets `offline: true` and maps that entry's model folder and filename directly from the descriptor's `domain` and `name`. Other entries are still resolved and queued for download.

URL formats accepted in `huggingface.URL`:

| Format | Example |
| --- | --- |
| Full HTTPS URL | `https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx/tree/main/cpu_and_mobile/...` |
| Short `user/repo` path | `ONNX-models/all-MiniLM-L6-v2-ONNX` |

When `huggingface.folder` is a `4D.Folder` the entire repository tree is fetched recursively and all file entries are added to `files`. When it is a `4D.File` only the matching single file entry is resolved.

In offline mode the following domain-to-option mappings are applied directly:

| domain | options key (folder) | options key (filename) |
| --- | --- | --- |
| `"chat.completion"` | `chat_completion_model` | — |
| `"embedding"` | `embeddings_model` | `embeddings_model_name` |
| `"rerank"` | `rerank_model` | `rerank_model_name` |
| `"systemone"` | `decision_model` | `decision_model_name` |
| `"tts"` | `tts_model` | `tts_model_name` |

For `systemone`, use a `4D.Folder` pointing to the local repository root and a selected ONNX `name` relative to it (for example `onnx/model_q4.onnx` for Kev or `onnx/decoder_model_merged_quantized.onnx` for d1). The folder-based recursive download preserves the entire repository layout, including root `config.json` and `tokenizer.json`, sibling ONNX graphs, and external data beside the selected graph. A single-file descriptor is insufficient for these models. Offline mapping uses the same root and name without requiring metadata downloads.

For native Strands Decider v1, use `opticsWolf/strands-decider-2B-qwen3.5-v1-2610-onnx` (Apache-2.0), an empty descriptor path, domain `systemone`, and selected name `onnx/model_q8.onnx` (CPU recommended). The unchanged recursive rules include root `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `onnx/model_q8.onnx_data` alongside the selected graph. They also download the other precisions: `name` is not a download filter. Offline use requires the selected graph, its external weights, and these root assets to be present already. See the [4D example and fidelity caveats](ONNX.md#strands-decider-v1-q8-cpu-recommended).

### Properties

| Property | Type | Description |
| --- | --- | --- |
| huggingfaces | cs.event.huggingfaces | Original descriptor collection |
| files | Collection | Resolved file metadata objects pending download |
| options | Object | Merged server options |
| event | cs.event.event | Callback functions |
| offline | Boolean | `True` when the Hugging Face API could not be reached for at least one entry |

### Methods

#### download ()

If `files` is empty, calls `_onNothingToDownload()` instead (the base implementation calls `start()`; see [`_Model`](_Model.md)). Otherwise gives each entry its own `downloadId`, then iterates `files`. For each file already present on disk with the correct size, `onDownload` is called immediately (skipping the HTTP fetch). Otherwise a `cs.event.download` is created to fetch the file from Hugging Face; its completion calls `onDownload(oid; downloadId)`.

#### onDownload (oid : Text; downloadId : Text)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| oid | Text | -> | OID of the file that finished downloading |
| downloadId | Text | -> | Destination identity assigned by `download()`; when omitted, every entry with `oid` is removed |

Removes the completed destination from `files`. Files with the same OID at different destinations are tracked separately. When this empties `files` (all downloads done) it calls `start()`; a late or repeated completion does not.

#### _notifyError (error : cs.event.error)

Calls `event.onError` with receiver `This` and arguments `(options; error)` when `event` is not `Null` and `onError` is a function.

#### _onNothingToDownload () *(virtual)*

Called by `download()` when there are no files to download. The base implementation calls `start()`. Overridden in [`_Model`](_Model.md).

#### models () → cs.event.models *(virtual)*

Returns a `cs.event.models` instance. The base implementation returns an empty collection. Overridden in [`_Model`](_Model.md).

#### start () *(virtual)*

Called automatically when all downloads complete. The base implementation is a no-op. Overridden in [`_Model`](_Model.md) to launch the server.

## See also

- [`_Model`](_Model.md) — concrete subclass that overrides `models()`, `onDownload()`, and `start()`
- [`ONNX`](ONNX.md) — the public entry point
