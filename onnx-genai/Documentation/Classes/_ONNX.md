# _ONNX
### Extends `_CLI` to target a named `onnx-genai` executable.

> _ONNX.new (command : Text; class : 4D.Class)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| command | Text | -> | Executable name passed to `_CLI` (e.g. `"ONNX-genai"`) |
| class | 4D.Class | -> | Optional custom controller class (must extend `_ONNX_Controller`) |

## Description

`_ONNX` extends [`_CLI`](_CLI.md) and passes `command` directly as the executable name to the parent constructor. It also walks the inheritance chain of the supplied `class` to decide whether to use it as a custom controller or fall back to the default `_ONNX_Controller`.

Unlike [`_CTranslate2`](../CTranslate2/_CTranslate2.md), which hardcodes `"ct2-server"`, `_ONNX` accepts the executable name as a parameter, allowing subclasses to target different `onnx-genai` variants. [`_server`](_server.md) passes `"ONNX-genai"`.

`_ONNX` is extended by [`_server`](_server.md) and should not be instantiated directly.

### Properties

In addition to properties inherited from `_CLI`:

| Property | Type | Description |
| --- | --- | --- |
| port | Integer | Port the server is listening on |
| onData | 4D.Function | Forwarded to the controller's `onData` handler |
| onDataError | 4D.Function | Forwarded to the controller's `onDataError` handler |
| onTerminate | 4D.Function | Called by `_ONNX_Controller` when the worker terminates |

### Methods

#### bind (option : Object; properties : Collection) → cs.ONNX._CLI

Copies listed property names from `option` into `This`. Used to bind event callbacks from an options object before execution.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| option | Object | -> | Source object |
| properties | Collection | -> | Property names to copy |
| Result | cs.ONNX._CLI | <- | `This` |

#### get worker () → 4D.SystemWorker

Returns the active `4D.SystemWorker` from the attached controller.

#### terminate ()

Delegates to `controller.terminate()`, stopping the active worker and draining the command queue.

## See also

- [`_CLI`](_CLI.md) — parent class
- [`_ONNX_Controller`](_ONNX_Controller.md) — default controller
- [`_server`](_server.md) — extends `_ONNX` with `start()`
