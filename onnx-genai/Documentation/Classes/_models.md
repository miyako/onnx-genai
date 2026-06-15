# _models
### Abstract base class that resolves Hugging Face model metadata and manages file downloads.

> _models.new (port : Integer; huggingfaces : cs.event.huggingfaces; options : Object; formula : 4D.Function; event : cs.event.event)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| port | Integer | -> | Port stored in `options.port` |
| huggingfaces | cs.event.huggingfaces | -> | One or more Hugging Face model descriptors |
| options | Object | -> | Server options object (mutated to add `port`, `onTerminate`, `onStdErr`, `onStdOut`) |
| formula | 4D.Function | -> | Internal response callback forwarded to the download helper |
| event | cs.event.event | -> | Callback functions |

## Description

`_models` is the download-and-startup orchestrator. On construction it iterates the `huggingfaces` collection, queries the Hugging Face API to resolve file and directory metadata, and builds an internal `files` queue. If the network is unreachable (`request.response.status = Null`) the class sets `offline: true`, maps the model folder and filename directly from the descriptor's `domain` and `name`, and skips downloading.

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
| `"tts"` | `tts_model` | `tts_model_name` |

### Properties

| Property | Type | Description |
| --- | --- | --- |
| huggingfaces | cs.event.huggingfaces | Original descriptor collection |
| files | Collection | Resolved file metadata objects pending download |
| options | Object | Merged server options |
| event | cs.event.event | Callback functions |
| offline | Boolean | `True` when the Hugging Face API could not be reached |

### Methods

#### download ()

Iterates `files`. For each file already present on disk with the correct size, `onDownload` is called immediately (skipping the HTTP fetch). Otherwise a `cs.event.download` is created to fetch the file from Hugging Face.

#### onDownload (oid : Text)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| oid | Text | -> | OID of the file that finished downloading |

Removes the completed file from `files`. When `files` is empty (all downloads done) it calls `start()`.

#### models () → cs.event.models *(virtual)*

Returns a `cs.event.models` instance. The base implementation returns an empty collection. Overridden in [`_Model`](_Model.md).

#### start () *(virtual)*

Called automatically when all downloads complete. The base implementation is a no-op. Overridden in [`_Model`](_Model.md) to launch the server.

## See also

- [`_Model`](_Model.md) — concrete subclass that overrides `models()`, `onDownload()`, and `start()`
- [`ONNX`](ONNX.md) — the public entry point
