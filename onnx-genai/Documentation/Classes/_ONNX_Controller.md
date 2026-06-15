# _ONNX_Controller
### Extends `_CLI_Controller` with `onTerminate` forwarding for the `onnx-genai` process.

> _ONNX_Controller.new (CLI : cs.ONNX._CLI)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| CLI | cs.ONNX._CLI | -> | The owning `_CLI` instance |

## Description

`_ONNX_Controller` is the default controller used by [`_ONNX`](_ONNX.md). It inherits all command-queue and worker-management behaviour from [`_CLI_Controller`](_CLI_Controller.md).

`onData`, `onDataError`, `onResponse`, and `onError` are intentionally left as the inherited no-op `_onEvent` handler. Only `onTerminate` is overridden, forwarding the termination event to the owning `_server` instance's `onTerminate` callback. This reflects `onnx-genai`'s role as a long-running HTTP server rather than a command with parseable incremental output.

### Overridden event callbacks

#### onTerminate ($worker : 4D.SystemWorker; $params : Object)

Called when the managed `SystemWorker` terminates. Looks up `onTerminate` on the owning `_server` instance and calls it if present.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| $worker | 4D.SystemWorker | -> | The worker that terminated |
| $params | Object | -> | Termination parameters from the system worker |

## See also

- [`_CLI_Controller`](_CLI_Controller.md) — parent class
- [`_ONNX`](_ONNX.md) — attaches this controller by default
- [`_server`](_server.md) — the `_ONNX` subclass whose `onTerminate` is forwarded here
