"""Compare HTTP decisions with independent CPU ONNX readout (numpy, tokenizers, onnxruntime)."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from systemone_http_test import call, request


def probabilities(scores):
    values = np.asarray(scores, dtype=np.float64)
    values = np.exp(values - values.max())
    return values / values.sum()


def verify(actual, expected):
    for name, distribution in expected.items():
        answer = actual["answers"][name]
        values = [answer["noul"]] if answer["type"] == "noul" else list(
            answer["probabilities"].values()
        )
        reference = [distribution["true"]] if answer["type"] == "noul" else [
            distribution[key] for key in answer["probabilities"]
        ]
        delta = float(np.max(np.abs(np.asarray(values) - reference)))
        assert delta < 1e-4, (name, delta, values, reference)
        print(f"{name}: max probability delta {delta:.8f}")


def main(graph, base, backend):
    graph = Path(graph)
    root = graph.parent.parent if graph.parent.name == "onnx" else graph.parent
    config = json.loads((root / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    model = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    payload = request(f"{backend}-latest")
    expected = {}
    total_tokens = 0

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    if backend == "kev":
        metadata = config["kev"]["delimiter_ids"]
        ids = [metadata["state"]] + encode(payload["state"])
        groups = {}
        for name, question in payload["questions"].items():
            ids += [metadata["question"]] + encode(question["instructions"])
            if question["type"] == "noul":
                choices = {"false": "no", "true": "yes"}
            elif question["type"] == "choice":
                choices = {
                    key: key + ": " + value for key, value in question["criteria"].items()
                }
            else:
                choices = {str(i): value for i, value in enumerate(question["criteria"])}
            ends = {}
            for key, text in choices.items():
                text = re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)
                ids += [metadata["option_start"]] + encode(text)
                ends[key] = len(ids)
                ids.append(metadata["option_end"])
            ids.append(metadata["decide"])
            groups[name] = ends
        tokens = np.asarray([ids], dtype=np.int64)
        logits = model.run(["logits"], {
            "input_ids": tokens, "attention_mask": np.ones_like(tokens)
        })[0][0]
        for name, ends in groups.items():
            expected[name] = dict(zip(ends, probabilities([logits[i] for i in ends.values()])))
        total_tokens = len(ids)
    else:
        suffix = graph.stem.removeprefix("decoder_model_merged")
        embedding = ort.InferenceSession(
            str(graph.parent / f"embed_tokens{suffix}.onnx"), options,
            providers=["CPUExecutionProvider"]
        )
        for name, question in payload["questions"].items():
            kind = question["type"]
            instruction = question["instructions"]
            if kind == "noul":
                question_text = instruction + "\n\nReply with yes or no only."
                groups = {
                    "false": [encode(s)[0] for s in ("no", "No", "NO") if len(encode(s)) == 1],
                    "true": [encode(s)[0] for s in ("yes", "Yes", "YES") if len(encode(s)) == 1],
                }
            elif kind == "choice":
                codes = ["A", "B"]
                lines = [
                    f"{code} {description}"
                    for code, description in zip(codes, question["criteria"].values())
                ]
                question_text = instruction + "\n\nOptions:\n" + "\n".join(lines)
                question_text += "\n\nReply with the option code only."
                groups = {
                    key: [encode(s)[0] for s in (code, " " + code) if len(encode(s)) == 1]
                    for key, code in zip(question["criteria"], codes)
                }
            else:
                lines = [f"{i} {value}" for i, value in enumerate(question["criteria"])]
                question_text = instruction + "\n\n" + "\n".join(lines)
                question_text += "\n\nReply with a single digit 0-2 only."
                groups = {str(i): [encode(str(i))[0]] for i in range(3)}
            prompt = (
                "<|startoftext|><|im_start|>user\n" + payload["state"]
                + "\n\n\nQUESTION:\n" + question_text + "<|im_end|>\n<|im_start|>assistant\n"
            )
            ids = np.asarray([encode(prompt)], dtype=np.int64)
            total_tokens += ids.size
            inputs = {
                "inputs_embeds": embedding.run(None, {"input_ids": ids})[0],
                "attention_mask": np.ones_like(ids),
            }
            text_config = config["text_config"]
            for item in model.get_inputs():
                if item.name.startswith("past_conv"):
                    inputs[item.name] = np.zeros(
                        (1, text_config["conv_dim"], text_config["conv_L_cache"]), np.float32
                    )
                elif item.name.startswith("past_key_values"):
                    inputs[item.name] = np.zeros((
                        1, text_config["num_key_value_heads"], 0,
                        text_config["hidden_size"] // text_config["num_attention_heads"]
                    ), np.float32)
            logits = model.run(["logits"], inputs)[0][0, -1]
            scores = [max(float(logits[i]) for i in group) for group in groups.values()]
            expected[name] = dict(zip(groups, probabilities(scores)))
    status, actual = call(base, "/v1/systemone", payload)
    assert status == 200, actual
    assert actual["usage"]["input_tokens"] == total_tokens, (actual["usage"], total_tokens)
    verify(actual, expected)
    print(f"{backend} parity passed (Python ONNX Runtime {ort.__version__}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["kev", "d1"])
    parser.add_argument("graph")
    parser.add_argument("base_url")
    arguments = parser.parse_args()
    main(arguments.graph, arguments.base_url, arguments.backend)
