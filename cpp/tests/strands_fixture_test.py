"""Synthetic Strands CLI/HTTP checks; usage: python strands_fixture_test.py BINARY.

Requires test-only numpy, onnx and tokenizers. No model download is needed.
Fixtures live in a TemporaryDirectory beneath the working directory.
"""

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

from systemone_http_test import call


ALIAS = "strands-decider-latest"
SCALE = np.float32(0.03125)
DEFAULT_CONFIG = {
    "architectures": ["StrandsDeciderModel"],
    "model_type": "qwen3_5_text",
    "decider": {
        "max_length": 4096,
        "temperature": 1.7,
        "temperature_by_kind": {"noul": 0.8, "choice": 1.3, "score": 2.1},
    },
}


def make_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(
        vocab={symbol: index for index, symbol in enumerate(alphabet)}, merges=[]
    ))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def make_graph(path, variant=None):
    shapes = {
        "input_ids": ["B", "L"], "attention_mask": ["B", "L"],
        "answer_pos": ["B"], "option_pos": ["B", "K"],
    }
    if variant == "input-rank":
        shapes["answer_pos"] = ["B", "L"]
    if variant == "static-input":
        shapes["input_ids"] = [1, "L"]
    inputs = []
    for name, shape in shapes.items():
        if variant == "missing-input" and name == "answer_pos":
            continue
        dtype = TensorProto.INT32 if variant == "input-dtype" and name == "input_ids" else TensorProto.INT64
        actual_name = "wrong_ids" if variant == "input-name" and name == "input_ids" else name
        inputs.append(helper.make_tensor_value_info(actual_name, dtype, shape))
    output_name = "wrong_logits" if variant == "output-name" else "logits"
    output_dtype = TensorProto.DOUBLE if variant == "output-dtype" else TensorProto.FLOAT
    output_shape = ["B", "K"]
    if variant == "output-rank":
        output_shape = ["B"]
    elif variant == "static-output":
        output_shape = [1, "K"]
    nodes = [
        helper.make_node("Cast", ["option_pos"], ["positions"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["positions", "scale"], ["scaled"]),
    ]
    if variant == "output-rank":
        nodes.append(helper.make_node("ReduceSum", ["scaled", "axes"], [output_name], keepdims=0))
    elif variant == "output-dtype":
        nodes.append(helper.make_node("Cast", ["scaled"], [output_name], to=TensorProto.DOUBLE))
    else:
        nodes.append(helper.make_node("Identity", ["scaled"], [output_name]))
    outputs = [helper.make_tensor_value_info(output_name, output_dtype, output_shape)]
    if variant == "extra-output":
        nodes.append(helper.make_node("Identity", ["scaled"], ["extra"]))
        outputs.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, ["B", "K"]))
    initializers = [numpy_helper.from_array(np.array(SCALE), "scale")]
    if variant == "output-rank":
        initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"))
    graph = helper.make_graph(nodes, "synthetic-strands", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def fixture(directory, name, config=None, variant=None):
    root = directory / name
    (root / "onnx").mkdir(parents=True)
    tokenizer = make_tokenizer()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    (root / "config.json").write_text(json.dumps(DEFAULT_CONFIG if config is None else config))
    graph = root / "onnx" / "model_q8.onnx"
    make_graph(graph, variant)
    return graph, tokenizer


def payload():
    return {
        "model": ALIAS,
        "state": "  A cafe\u0301 customer requests a refund.  ",
        "questions": {
            "truth": {"type": "noul", "instructions": "Does the customer want help?"},
            "team": {
                "type": "choice", "instructions": "Select a team.",
                "criteria": {"zebra": "  Billing\tand\nrefunds ", "alpha": "Technical support"},
            },
            "urgency": {
                "type": "score", "instructions": "Rate urgency.",
                "criteria": ["Low", "Moderate", "Very high"],
            },
        },
    }


def content(value):
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def reference(tokenizer, state, question, metadata):
    """Derive readout indices from independently tokenized option-line prefixes."""
    kind = question["type"]
    headers = {
        "noul": "Decide whether the statement is true of the state.",
        "choice": "Select exactly one option.",
        "score": "Rate the state against the ordered levels below (lowest first).",
    }
    criteria = question.get("criteria")
    keys = ["false", "true"] if kind == "noul" else (
        list(criteria) if kind == "choice" else list(map(str, range(len(criteria))))
    )
    state_length = len(tokenizer.encode(f"<state>\n{content(state)}\n</state>\n").ids)
    prompt = f'<question type="{kind}">\n{headers[kind]}\n{content(question.get("instructions"))}\n<options>\n'
    positions = []
    for index, key in enumerate(keys):
        if kind == "noul":
            description = (criteria or {}).get(
                key, "the statement does not hold for this state" if key == "false"
                else "the statement holds for this state"
            )
        else:
            description = criteria[key] if kind == "choice" else criteria[index]
        description = " ".join(content(description).split())
        prompt += f"{index + 1}. {key}" + (f" — {description}" if description else "")
        positions.append(state_length + len(tokenizer.encode(prompt).ids) - 1)
        prompt += "\n"
    prompt += "</options>\n</question>\n<answer>"
    temperature = metadata.get("temperature_by_kind", {}).get(kind, metadata.get("temperature"))
    logits = np.asarray(positions, dtype=np.float32) * SCALE
    logits = logits.astype(np.float64) / temperature
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return keys, probabilities, state_length + len(tokenizer.encode(prompt).ids)


def check_response(body, request, tokenizer, config, model_id):
    assert set(body) == {"model", "answers", "usage"}, body
    assert body["model"] == model_id, body
    assert set(body["answers"]) == set(request["questions"]), body
    tokens = 0
    for name, question in request["questions"].items():
        keys, expected, length = reference(tokenizer, request["state"], question, config["decider"])
        tokens += length
        answer = body["answers"][name]
        assert answer["type"] == question["type"], answer
        if question["type"] == "noul":
            assert set(answer) == {"type", "noul"}, answer
            np.testing.assert_allclose(answer["noul"], expected[1], rtol=0, atol=1e-6)
            continue
        assert set(answer["probabilities"]) == set(keys), answer
        np.testing.assert_allclose([answer["probabilities"][key] for key in keys], expected, rtol=0, atol=1e-6)
        mode = int(expected.argmax())
        if question["type"] == "choice":
            assert answer["choice"] == keys[mode], answer
            confidence = 1 if len(keys) == 1 else (expected[mode] - 1 / len(keys)) / (1 - 1 / len(keys))
        else:
            assert answer["legend"] == dict(zip(keys, question["criteria"])), answer
            np.testing.assert_allclose(answer["score"], np.dot(np.arange(len(keys)), expected), rtol=0, atol=1e-6)
            spread = np.dot(expected, np.abs(np.arange(len(keys)) - mode))
            even_spread = np.mean(np.abs(np.arange(len(keys)) - (len(keys) - 1) / 2))
            confidence = np.clip(1 - spread / even_spread, 0, 1)
        np.testing.assert_allclose(answer["confidence"], confidence, rtol=0, atol=1e-6)
    assert body["usage"] == {"input_tokens": tokens, "output_tokens": 0}, body["usage"]


def cli(binary, graph, request, directory, success=True, diagnostic=None):
    input_path, output_path = directory / "request.json", directory / "response.json"
    input_path.write_text(json.dumps(request, ensure_ascii=False))
    output_path.unlink(missing_ok=True)
    result = subprocess.run(
        [str(binary), "-D", str(graph), "-i", str(input_path), "-o", str(output_path)],
        capture_output=True, text=True, timeout=60,
    )
    body = json.loads(output_path.read_text()) if output_path.exists() else None
    details = result.stdout + result.stderr + json.dumps(body)
    assert (result.returncode == 0) == success, details
    if diagnostic:
        assert diagnostic.lower() in details.lower(), details
    return body


@contextmanager
def server(binary, graph, directory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (directory / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [str(binary), "-s", "-D", str(graph), "-p", str(port)], stdout=log, stderr=log
        )
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(120):
                if process.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                try:
                    status, catalogue = call(base, "/v1/models", timeout=2)
                    assert status == 200, catalogue
                    yield base, catalogue
                    return
                except urllib.error.URLError:
                    time.sleep(0.25)
            raise AssertionError("Synthetic Strands server did not become ready.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def exercise(binary, directory):
    graph, tokenizer = fixture(directory, "synthetic-strands-v1")
    request = payload()
    model_id = graph.parent.parent.name
    for model in (ALIAS, model_id):
        current = dict(request, model=model)
        check_response(cli(binary, graph, current, directory), current, tokenizer, DEFAULT_CONFIG, model_id)
    with server(binary, graph, directory) as (base, catalogue):
        assert catalogue["object"] == "list" and len(catalogue["data"]) == 1, catalogue
        assert catalogue["data"][0]["id"] == model_id, catalogue
        assert catalogue["data"][0]["backend"] == "strands-decider", catalogue
        for model in (ALIAS, model_id):
            current = dict(request, model=model)
            status, body = call(base, "/v1/systemone", current)
            assert status == 200, body
            check_response(body, current, tokenizer, DEFAULT_CONFIG, model_id)
        for name, question in request["questions"].items():
            current = dict(request, questions={name: question})
            status, body = call(base, "/v1/systemone", current)
            assert status == 200, body
            check_response(body, current, tokenizer, DEFAULT_CONFIG, model_id)
        singleton = dict(request, questions={"one": {"type": "choice", "criteria": {"only": "One"}}})
        status, body = call(base, "/v1/systemone", singleton)
        assert status == 200, body
        check_response(body, singleton, tokenizer, DEFAULT_CONFIG, model_id)
        bad_requests = [
            (dict(request, model="wrong-model"), "model"),
            (dict(request, questions={"one": {"type": "score", "criteria": ["Only"]}}), "criteria"),
            (dict(request, state="x" * 4096), "4096"),
            (dict(request, state="x" * 700, questions={
                f"q{i}": request["questions"]["truth"] for i in range(40)
            }), "32768"),
        ]
        for bad, diagnostic in bad_requests:
            status, body = call(base, "/v1/systemone", bad)
            assert status == 422 and diagnostic in json.dumps(body), (bad, body)
            cli(binary, graph, bad, directory, success=False, diagnostic=diagnostic)

    for label, metadata in (
        ("temperature-fallback", {"max_length": 4096, "temperature": 1.7}),
        ("partial-fallback", {"max_length": 4096, "temperature": 1.7, "temperature_by_kind": {"choice": 0.9}}),
        ("per-kind-only", {"max_length": 4096, "temperature_by_kind": {"noul": 0.8, "choice": 1.3, "score": 2.1}}),
    ):
        config = {"decider": metadata}
        candidate, tok = fixture(directory, label, config)
        check_response(cli(binary, candidate, request, directory), request, tok, config, label)

    variants = [
        "missing-input", "input-name", "input-rank", "input-dtype", "static-input",
        "output-name", "output-rank", "output-dtype", "static-output", "extra-output",
    ]
    for variant in variants:
        candidate, _ = fixture(directory, variant, variant=variant)
        cli(binary, candidate, request, directory, success=False)

    configs = [
        ("no-max-length", {"decider": {"temperature": 1}}, "max_length"),
        ("zero-max-length", {"decider": {"max_length": 0, "temperature": 1}}, "max_length"),
        ("string-max-length", {"decider": {"max_length": "4096", "temperature": 1}}, "max_length"),
        ("no-temperature", {"decider": {"max_length": 4096}}, "temperature"),
        ("zero-temperature", {"decider": {"max_length": 4096, "temperature": 0}}, "temperature"),
        ("negative-temperature", {"decider": {"max_length": 4096, "temperature": -1}}, "temperature"),
        ("string-temperature", {"decider": {"max_length": 4096, "temperature": "1"}}, "temperature"),
        ("bad-kind-temperature", {"decider": {"max_length": 4096, "temperature": 1, "temperature_by_kind": {"score": 0}}}, "temperature"),
        ("bad-kind-map", {"decider": {"max_length": 4096, "temperature": 1, "temperature_by_kind": []}}, "temperature"),
        ("missing-backend", {}, "backend"),
        ("ambiguous-kev", dict(copy.deepcopy(DEFAULT_CONFIG), kev={}), "backend"),
        ("ambiguous-d1", dict(copy.deepcopy(DEFAULT_CONFIG), d1={}), "backend"),
        ("bad-decider", {"decider": []}, "metadata"),
    ]
    for label, config, diagnostic in configs:
        candidate, _ = fixture(directory, label, config)
        cli(binary, candidate, request, directory, success=False, diagnostic=diagnostic)
    print("Synthetic Strands dispatch, probability, HTTP/CLI, graph/config and limit checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="Rebuilt onnx-genai executable")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Rebuilt executable not found: {binary}")
    with tempfile.TemporaryDirectory(prefix="strands-fixture-", dir=Path.cwd()) as directory:
        exercise(binary, Path(directory))


if __name__ == "__main__":
    main()
