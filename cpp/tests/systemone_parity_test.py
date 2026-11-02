"""Compare HTTP decisions with independent CPU ONNX readout (numpy, tokenizers, onnxruntime)."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

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


def main(graph, base, backend, payload=None, preparer=None, threads=2, timeout=120):
    graph = Path(graph)
    root = graph.parent.parent if graph.parent.name == "onnx" else graph.parent
    config = json.loads((root / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    payload = payload or request(f"{backend}-latest")
    expected = {}
    total_tokens = 0
    prepared = None
    if preparer:
        assert backend == "strands-decider", "--preparer is for Strands marker-position checks"
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "request.json"
            fixture.write_text(json.dumps(payload, ensure_ascii=False))
            result = subprocess.run(
                [preparer, "--strands-inputs", str(graph), str(fixture)],
                check=True, capture_output=True, text=True, timeout=120
            )
            prepared = json.loads(result.stdout)
    model = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])

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
    elif backend == "strands-decider":
        def content(value):
            if value is None:
                return ""
            return value.strip() if isinstance(value, str) else json.dumps(
                value, indent=2, ensure_ascii=False
            )

        # Match the official engine's separate state/question tokenization, not
        # a concatenated encode (BPE can merge across the shared-prefix boundary).
        state_ids = encode("<state>\n" + content(payload["state"]) + "\n</state>\n")
        for name, question in payload["questions"].items():
            kind = question["type"]
            if kind == "noul":
                header = "Decide whether the statement is true of the state."
                criteria = {
                    "false": "the statement does not hold for this state",
                    "true": "the statement holds for this state",
                }
                criteria.update(question.get("criteria") or {})
            elif kind == "choice":
                header = "Select exactly one option."
                criteria = question["criteria"]
            else:
                header = "Rate the state against the ordered levels below (lowest first)."
                criteria = {str(i): v for i, v in enumerate(question["criteria"])}
            text = (
                f'<question type="{kind}">\n{header}\n'
                + content(question.get("instructions")) + "\n<options>\n"
            )
            spans = []
            for i, (key, value) in enumerate(criteria.items()):
                start = len(text)
                description = " ".join(content(value).split())
                text += f"{i + 1}. {key}" + (f" — {description}" if description else "")
                spans.append((start, len(text)))
                text += "\n"
            text += "</options>\n</question>\n<answer>"
            encoded = tokenizer.encode(text, add_special_tokens=False)
            option_positions = []
            for start, end in spans:
                surviving = [
                    i for i, (lo, hi) in enumerate(encoded.offsets)
                    if hi > lo and lo >= start and hi <= end
                ]
                assert surviving, (name, start, end)
                option_positions.append(len(state_ids) + surviving[-1])
            tokens = np.asarray([state_ids + encoded.ids], np.int64)
            if prepared is not None:
                assert prepared[name] == {
                    "input_ids": tokens[0].tolist(),
                    "option_pos": option_positions,
                    "answer_pos": tokens.size - 1,
                }, (name, "native token IDs or marker positions differ from Python offsets")
            total_tokens += tokens.size
            logits = model.run(["logits"], {
                "input_ids": tokens, "attention_mask": np.ones_like(tokens),
                "answer_pos": np.asarray([tokens.size - 1], np.int64),
                "option_pos": np.asarray([option_positions], np.int64),
            })[0][0]
            metadata = config["decider"]
            temperature = metadata.get("temperature_by_kind", {}).get(
                kind, metadata.get("temperature")
            )
            expected[name] = dict(zip(criteria, probabilities(logits / temperature)))
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
    del model
    status, actual = call(base, "/v1/systemone", payload, timeout=timeout)
    assert status == 200, actual
    assert actual["usage"]["input_tokens"] == total_tokens, (actual["usage"], total_tokens)
    verify(actual, expected)
    print(f"{backend} parity passed (Python ONNX Runtime {ort.__version__}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["kev", "d1", "strands-decider"])
    parser.add_argument("graph")
    parser.add_argument("base_url")
    parser.add_argument("--request", help="JSON request fixture, including structured/Unicode content")
    parser.add_argument("--preparer", help="Focused native test binary, for exact token IDs/positions")
    parser.add_argument("--threads", type=int, default=2, help="Match the server's Detected Intra-Op threads")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout for slow CPU runs")
    arguments = parser.parse_args()
    payload = json.loads(Path(arguments.request).read_text()) if arguments.request else None
    main(arguments.graph, arguments.base_url, arguments.backend, payload, arguments.preparer,
         arguments.threads, arguments.timeout)
