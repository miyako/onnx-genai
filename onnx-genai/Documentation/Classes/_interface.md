# _interface
### Abstract base class providing shared server lifecycle methods for the ONNX namespace.

> _interface.new ()

`_interface` has no constructor parameters. It is not instantiated directly; it is extended by [`ONNX`](ONNX.md).

## Description

`_interface` is the root class of the `cs.ONNX` hierarchy. It provides two methods used by all subclasses:

- **`_onTCP`** — a callback invoked after a TCP port-availability check. If the port is free it dispatches a worker to start the server; if the port is already in use it fires `event.onError` with a descriptive message.
- **`terminate`** — gracefully shuts down the running `onnx-genai` worker.

### _onTCP ($status, $options)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| $status | Object | -> | Result of the TCP port check (`success`, `port`, `PID`) |
| $options | Object | -> | Server options including `name`, `port`, `event` |

Called internally after a port probe. When `$status.success` is `true` the method calls `CALL WORKER` to invoke `start` on the server worker, then registers `onModel` as the response callback. When the port is occupied it constructs a `cs.event.error` and calls `event.onError`.

### terminate ()

Obtains the server worker via `cs.ONNX.workers.worker` and calls `terminate()` on it.

## Examples

```4d
var $onnx : cs.ONNX.ONNX
$onnx:=cs.ONNX.ONNX.new()
$onnx.terminate()
```

## See also

- [`ONNX`](ONNX.md) — public subclass that extends `_interface`
- [`_server`](_server.md) — CLI wrapper whose worker is targeted by `terminate()`
