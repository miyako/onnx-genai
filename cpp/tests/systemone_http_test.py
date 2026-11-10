"""Exercise a rebuilt server; pass decision graph arguments for real-model checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def call(base, path, payload=None, timeout=120):
    data = None if payload is None else (
        payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    )
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def request(model):
    return {
        "model": model,
        "state": "I was charged twice. Please refund the extra charge.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "technical": "Software faults"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["Routine", "Urgent", "Emergency"],
            },
        },
    }


def check_response(body, payload):
    assert set(body) == {"model", "answers", "usage"}, body
    assert set(body["answers"]) == set(payload["questions"]), body
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0
    for name, question in payload["questions"].items():
        answer = body["answers"][name]
        assert answer["type"] == question["type"]
        if question["type"] == "noul":
            assert set(answer) == {"type", "noul"}
            assert 0 <= answer["noul"] <= 1
            continue
        probabilities = answer["probabilities"]
        assert abs(sum(probabilities.values()) - 1) < 1e-6
        assert all(0 <= p <= 1 for p in probabilities.values())
        assert 0 <= answer["confidence"] <= 1
        if question["type"] == "choice":
            assert set(probabilities) == set(question["criteria"])
            assert answer["choice"] == max(probabilities, key=probabilities.get)
        else:
            assert answer["legend"] == {
                str(i): level for i, level in enumerate(question["criteria"])
            }
            assert abs(answer["score"] - sum(int(i) * p for i, p in probabilities.items())) < 1e-6


def exercise(binary, backend=None, graph=None, semantic=True, packed=False):
    binary = str(Path(binary).resolve())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [binary, "-s", "-p", str(port)]
    if graph:
        command += ["-D", graph]
    with tempfile.TemporaryFile(mode="w+", dir=Path.cwd()) as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                try:
                    status, catalogue = call(base, "/v1/models")
                    break
                except urllib.error.URLError:
                    time.sleep(0.5)
            else:
                raise AssertionError("Server did not become ready.")
            assert status == 200 and catalogue["object"] == "list"
            payload = request(f"{backend}-latest" if backend else "not-loaded")
            status, body = call(base, "/v1/systemone", payload)
            if not graph:
                assert catalogue["data"] == []
                assert status == 503 and body["error"]["type"] == "service_unavailable"
                for path in ("/v1/chat/completions", "/v1/embeddings", "/v1/rerank", "/v1/audio/speech"):
                    assert call(base, path, {})[0] == 400
            else:
                assert status == 200, body
                check_response(body, payload)
                assert catalogue["data"][0]["id"] == body["model"]
                assert catalogue["data"][0]["backend"] == backend
                if semantic:
                    assert body["answers"]["refund"]["noul"] > 0.5
                    assert body["answers"]["team"]["choice"] == "billing"
                isolation_tolerance = 1e-3 if backend == "kev" else 1e-4
                for name, question in payload["questions"].items():
                    single = dict(payload, questions={name: question})
                    single_status, isolated = call(base, "/v1/systemone", single)
                    assert single_status == 200, isolated
                    a, b = body["answers"][name], isolated["answers"][name]
                    if packed:
                        # LiquidAI d1 reads a state's questions in one "set" row: packing moves the
                        # probabilities by design, so only the decisions must agree.
                        if question["type"] == "noul":
                            assert (a["noul"] > 0.5) == (b["noul"] > 0.5), (name, a, b)
                        elif question["type"] == "choice":
                            assert a["choice"] == b["choice"], (name, a, b)
                    elif question["type"] == "noul":
                        assert abs(a["noul"] - b["noul"]) < isolation_tolerance
                    else:
                        assert all(
                            abs(p - b["probabilities"][key]) < isolation_tolerance
                            for key, p in a["probabilities"].items()
                        )
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(
                        lambda _: call(base, "/v1/systemone", dict(
                            payload, questions={"refund": payload["questions"]["refund"]}
                        )), range(4)
                    ))
                assert all(status == 200 for status, _ in results), results
                assert call(base, "/v1/systemone", dict(payload, model="wrong-model"))[0] == 422
                if backend == "laya":
                    for alias in ("kev-latest", "d1-latest"):
                        assert call(base, "/v1/systemone", dict(payload, model=alias))[0] == 422
                    too_wide = dict(payload, questions={
                        "q": {"type": "choice", "criteria": {
                            f"option_{i}": f"Candidate {i}" for i in range(255)
                        }}
                    })
                    status, error = call(base, "/v1/systemone", too_wide)
                    assert status == 422 and error["detail"][0]["loc"][-1] == "criteria", error
                oversized = dict(payload, state="token " * 9000)
                assert call(base, "/v1/systemone", oversized)[0] == 422
                with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                    input_path = Path(directory) / "request.json"
                    output_path = Path(directory) / "response.json"
                    input_path.write_text(json.dumps(payload))
                    cli = [binary, "-D", graph, "-i", str(input_path), "-o", str(output_path)]
                    completed = subprocess.run(cli, capture_output=True, text=True, timeout=120)
                    assert completed.returncode == 0, completed.stderr
                    check_response(json.loads(output_path.read_text()), payload)
                    input_path.write_text("{}")
                    completed = subprocess.run(cli, capture_output=True, text=True, timeout=120)
                    assert completed.returncode != 0
                    assert "detail" in json.loads(output_path.read_text())
                    graph_path = Path(graph).resolve()
                    relative_root = graph_path.parent.parent if graph_path.parent.name == "onnx" else graph_path.parent
                    relative = [binary, "-D", str(graph_path.relative_to(relative_root)),
                                "-i", str(input_path), "-o", str(output_path)]
                    input_path.write_text(json.dumps(payload))
                    completed = subprocess.run(
                        relative, cwd=relative_root,
                        capture_output=True, text=True, timeout=120
                    )
                    assert completed.returncode == 0, completed.stderr
                    relative_body = json.loads(output_path.read_text())
                    assert relative_body["model"] == body["model"], relative_body
                    missing_output = [binary, "-D", graph, "-i", str(input_path),
                                      "-o", str(Path(directory) / "missing" / "response.json")]
                    completed = subprocess.run(missing_output, capture_output=True, text=True, timeout=120)
                    assert completed.returncode != 0 and "output file" in completed.stderr
            invalid = [
                "not json", "{} {}", '{"model":"x","model":"duplicate"}', "{}",
                dict(payload, state=None),
                dict(payload, images=["unsupported"]),
                dict(payload, assets={"photo": {}}),
                dict(payload, questions={"box": {"type": "box", "instructions": "Find it"}}),
                dict(payload, questions={"q": {"type": "choice", "criteria": {}}}),
                dict(payload, questions={"q": {"type": "score", "criteria": ["x"] * 11}}),
                dict(payload, questions={"q": {"type": "noul", "instructions": ""}}),
            ]
            for bad in invalid:
                status, error = call(base, "/v1/systemone", bad)
                assert status == 422 and error["detail"][0]["loc"][0] == "body", (bad, error)
            print(f"{backend or 'no-model'} HTTP/CLI checks passed.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--kev")
    parser.add_argument("--d1")
    parser.add_argument("--lfm3", help="LiquidAI d1-2B / d1-Omni-2B decoder graph (served as d1)")
    parser.add_argument("--laya")
    parser.add_argument("--fixtures", type=Path, help="Generated tiny Laya/Kev/d1 fixture root (no semantic quality assertions)")
    parser.add_argument("--strands", help="Current v1 fused Strands Decider graph")
    args = parser.parse_args()
    exercise(args.binary)
    for backend, graph in (("kev", args.kev), ("d1", args.d1), ("laya", args.laya),
                           ("strands-decider", args.strands)):
        if graph:
            exercise(args.binary, backend, graph)
    if args.lfm3:
        exercise(args.binary, "d1", args.lfm3, packed=True)
    if args.fixtures:
        root = args.fixtures.resolve()
        for backend, graph in (
            ("laya", root / "model.onnx"),
            ("kev", root / "dispatch-kev" / "model.onnx"),
            ("d1", root / "dispatch-d1" / "decoder_model_merged.onnx"),
        ):
            exercise(args.binary, backend, str(graph), semantic=False)
