"""Laya native/ORT parity, plus explicitly requested official PyTorch/ONNX comparison."""

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from systemone_http_test import call, check_response


def render(value, ascii=False):
    return json.dumps(value, ensure_ascii=ascii, separators=(", ", ": "))


def content(value):
    return value if isinstance(value, str) else render(value)


def sequence(payload, question, tokenizer, config):
    def encode(text):
        return tokenizer.encode(text.replace("[MASK]", " "), add_special_tokens=False).ids

    kind = question["type"]
    # Missing and explicit null instructions both serialize as JSON null, not "".
    instruction = question.get("instructions")
    head = encode(kind + " question: " + (
        instruction if isinstance(instruction, str) else render(instruction, ascii=True)
    ))
    criteria = question.get("criteria", {})
    if kind == "choice":
        keys = list(criteria)
        options = [key if value is None or value == "" else key + ": " + content(value)
                   for key, value in criteria.items()]
    elif kind == "score":
        keys = [str(i) for i in range(len(criteria))]
        options = [f"level {i}: " + content(value) for i, value in enumerate(criteria)]
    else:
        keys = ["false", "true"]
        options = [
            "false: " + (content(criteria["false"]) if criteria.get("false") not in (None, "")
                         else "no, the statement does not hold"),
            "true: " + (content(criteria["true"]) if criteria.get("true") not in (None, "")
                        else "yes, the statement holds"),
        ]
    encoded = [encode(" " + option) for option in options]
    assert all(len(option) <= 48 for option in encoded), "Reference option exceeds 48 tokens"
    option_size = sum(len(option) + 1 for option in encoded)
    assert option_size <= config["head_max_len"], "Reference head options overflow"
    budget = config["head_max_len"] - option_size
    if budget < 16:
        per = max(4, (config["head_max_len"] - 16) // len(encoded))
        assert all(len(option) + 1 <= per for option in encoded), "Reference requires truncation"
    assert len(head) <= max(8, budget), "Reference instruction overflow"
    cls, sep, mask, _ = [tokenizer.token_to_id(token)
                         for token in ("[CLS]", "[SEP]", "[MASK]", "[PAD]")]
    ids = [cls] + head + [sep]
    positions = []
    for option in encoded:
        positions.append(len(ids))
        ids += [mask] + option
    ids += [sep] + encode(content(payload["state"])) + [sep]
    assert len(ids) <= config["max_len"], "Reference sequence overflow"
    return ids, positions, ["choice", "score", "noul"].index(kind), keys


def reference(payload, tokenizer, config, session):
    rows = [sequence(payload, q, tokenizer, config) for q in payload["questions"].values()]
    length = max(len(row[0]) for row in rows)
    width = max(2, max(len(row[1]) for row in rows))
    assert length * len(rows) <= 32768, "Reference padded-token overflow"
    ids = np.full((len(rows), length), tokenizer.token_to_id("[PAD]"), dtype=np.int64)
    mask = np.zeros_like(ids)
    positions = np.zeros((len(rows), width), dtype=np.int64)
    markers = np.zeros_like(positions, dtype=bool)
    types = np.empty(len(rows), dtype=np.int64)
    for row, (tokens, marks, kind, _) in enumerate(rows):
        ids[row, :len(tokens)] = tokens
        mask[row, :len(tokens)] = 1
        positions[row, :len(marks)] = marks
        markers[row, :len(marks)] = True
        types[row] = kind
    logits, acts = session.run(["logits", "act_probs"], {
        "input_ids": ids, "attention_mask": mask, "marker_pos": positions,
        "marker_mask": markers, "qtype": types,
    })
    assert logits.shape == (len(rows), width) and acts.shape == (len(rows), 2)
    assert np.isfinite(acts).all() and (acts >= 0).all() and (acts <= 1).all()
    assert np.max(np.abs(acts.sum(axis=1) - 1)) < 1e-5
    answers = {}
    for row, (name, question) in enumerate(payload["questions"].items()):
        _, _, kind, keys = rows[row]
        count = len(keys)
        bucket = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
        temperature = config.get("temperature_by_options", {}).get(
            question["type"] + ":" + bucket, config["temperature"][kind]
        )
        values = logits[row, :count].astype(np.float64) / temperature
        values = np.exp(values - values.max())
        answers[name] = dict(zip(keys, values / values.sum()))
    return answers, sum(len(row[0]) for row in rows)


def payloads(model):
    base = {
        "model": model, "state": {
            "z": "state [MASK] é", "a": ["second", True],
            "floats": [0.1, 1.0, -0.0, 1e-4, 1e-5, 1e15, 1e16],
        },
        "questions": {
            "choice": {"type": "choice", "instructions": {
                           "z": "choose é [MASK]", "floats": [0.1, -0.0, 1e-5], "del": "\x7f",
                       },
                       "criteria": {"beta": None, "alpha": {"label": "one"}, "é": ""}},
            "noul": {"type": "noul", "instructions": "true? [MASK]"},
            "score": {"type": "score", "instructions": "rate",
                      "criteria": ["low", {"middle": "é"}, "high"]},
            "singleton": {"type": "choice", "criteria": {"one": None}},
            "noul_custom": {"type": "noul", "criteria": {"false": "low", "true": {"a": "high"}}},
        },
    }
    yield base
    plain = deepcopy(base)
    plain["state"] = "state " * 17
    plain["questions"]["choice"]["instructions"] = "choose"
    yield plain
    for count in (2, 5, 6, 10, 11):
        yield dict(base, state="state", questions={
            "wide": {"type": "choice", "instructions": "choose",
                     "criteria": {f"option{i}": None for i in range(count)}},
            "other": {"type": "noul", "instructions": "true?"},
        })


def verify(actual, expected, tolerance):
    for name, distribution in expected.items():
        answer = actual["answers"][name]
        if answer["type"] == "noul":
            assert abs(answer["noul"] - distribution["true"]) <= tolerance, (name, answer, distribution)
            continue
        assert set(answer["probabilities"]) == set(distribution)
        for key, value in distribution.items():
            assert abs(answer["probabilities"][key] - value) <= tolerance, (name, key, answer, distribution)
        values = np.array(list(distribution.values()))
        if answer["type"] == "choice":
            confidence = 1.0 if len(values) == 1 else (
                float(values.max()) - 1 / len(values)
            ) / (1 - 1 / len(values))
        else:
            mean = sum(int(key) * value for key, value in distribution.items())
            assert abs(answer["score"] - mean) <= tolerance
            mode = int(np.argmax(values))
            even_spread = sum(abs(i - (len(values) - 1) / 2) for i in range(len(values))) / len(values)
            confidence = 1.0 if len(values) == 1 else 1 - (
                sum(abs(int(key) - mode) * value for key, value in distribution.items())
                / even_spread
            )
        confidence = min(1.0, max(0.0, confidence))
        assert abs(answer["confidence"] - confidence) <= tolerance, (name, answer["confidence"], confidence)


def main(graph, base, tolerance):
    graph = Path(graph).resolve()
    root = graph.parent
    if root.name == "onnx" and not (root / "laya_config.json").exists():
        root = root.parent
    config = json.loads((root / "laya_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    for payload in payloads("laya-latest"):
        expected, tokens = reference(payload, tokenizer, config, session)
        status, actual = call(base, "/v1/systemone", payload)
        assert status == 200, actual
        check_response(actual, payload)
        assert actual["usage"]["input_tokens"] == tokens, (actual["usage"], tokens)
        verify(actual, expected, tolerance)
        for name, question in payload["questions"].items():
            isolated = dict(payload, questions={name: question})
            reference_single, single_tokens = reference(isolated, tokenizer, config, session)
            status, response = call(base, "/v1/systemone", isolated)
            assert status == 200, response
            assert response["usage"]["input_tokens"] == single_tokens
            verify(response, reference_single, tolerance)
            verify(response, {name: expected[name]}, tolerance)
    print(f"Laya parity passed (Python ONNX Runtime {ort.__version__}, tolerance {tolerance:g}).")


def compare_pytorch(graph, directory, logit_tolerance, act_tolerance):
    """Run only on explicit request: this loads the full official checkpoint on CPU."""
    directory = Path(directory).resolve()
    graph = Path(graph).resolve()
    root = graph.parent
    if root.name == "onnx" and not (root / "laya_config.json").exists():
        root = root.parent
    for path in (directory / "rl_common.py", directory / "rl_agent_config.json",
                 directory / "model.safetensors", directory / "encoder" / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing supplied official PyTorch comparison file: {path}")
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "--pytorch-dir additionally requires torch, transformers and safetensors "
            "in the test environment; ordinary native/ORT parity does not."
        ) from error
    spec = importlib.util.spec_from_file_location("laya_official_reference", directory / "rl_common.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    torch.set_num_threads(2)
    config = json.loads((directory / "rl_agent_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    model = official.build_model(config, encoder_dir=str(directory / "encoder"))
    model.load_state_dict(load_file(str(directory / "model.safetensors"), device="cpu"), strict=True)
    model.eval()
    model.encoder.config.reference_compile = False
    state = {"message": "I was charged twice. Please refund the extra charge.", "language": "café"}
    questions = [
        {"t": "noul", "ins": "Is the customer requesting a refund?", "crit": None},
        {"t": "choice", "ins": "Which team should handle this?",
         "crit": {"billing": "Payments and refunds", "technical": "Software faults"}},
        {"t": "score", "ins": "How urgent is this request?", "crit": ["Routine", "Urgent", "Emergency"]},
        {"t": "choice", "ins": "Which number is largest?", "crit": {str(i): str(i) for i in range(12)}},
    ]
    items = []
    for question in questions:
        ids, markers = official.build_sequence(
            tokenizer, state, question, config["max_len"], config["head_max_len"]
        )
        items.append({
            "ids": ids, "markers": markers, "qtype": official.QTYPES[question["t"]],
            "target": [0.0] * len(markers), "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1,
        })
    batch = official.collate_items([items], tokenizer.pad_token_id)
    names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    with torch.inference_mode():
        logits, action_logits = model(*(batch[name] for name in names))
        expected_logits = logits.cpu().numpy()
        expected_actions = torch.softmax(action_logits, -1).cpu().numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    actual_logits, actual_actions = session.run(
        ["logits", "act_probs"], {name: batch[name].numpy() for name in names}
    )
    assert expected_logits.shape == actual_logits.shape
    assert expected_actions.shape == actual_actions.shape
    logit_delta = float(np.max(np.abs(expected_logits - actual_logits)))
    act_delta = float(np.max(np.abs(expected_actions - actual_actions)))
    print(json.dumps({
        "torch": torch.__version__, "onnxruntime": ort.__version__,
        "max_raw_logit_difference": logit_delta, "max_act_probability_difference": act_delta,
        "batch": len(items), "lengths": [len(item["ids"]) for item in items],
        "calibration_compared": False,
    }), flush=True)
    assert logit_delta < logit_tolerance, (logit_delta, logit_tolerance)
    assert act_delta < act_tolerance, (act_delta, act_tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", help="Generated fixture or optional real Laya graph")
    parser.add_argument("base_url", nargs="?", help="Running native server URL")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--pytorch-dir", type=Path,
                        help="Opt in to full CPU checkpoint comparison using supplied official rl_common.py/config/weights/encoder")
    parser.add_argument("--pytorch-logit-tolerance", type=float, default=1e-4)
    parser.add_argument("--pytorch-act-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if not args.base_url and not args.pytorch_dir:
        parser.error("Provide a native server base_url and/or explicitly opt in with --pytorch-dir")
    if args.base_url:
        main(args.graph, args.base_url, args.tolerance)
    if args.pytorch_dir:
        compare_pytorch(args.graph, args.pytorch_dir, args.pytorch_logit_tolerance, args.pytorch_act_tolerance)
