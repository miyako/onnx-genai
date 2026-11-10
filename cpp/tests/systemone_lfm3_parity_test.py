"""LiquidAI d1 (d1-2B-ONNX, d1-Omni-2B-ONNX) parity: HTTP decisions against an independent CPU readout.

Independently rebuilds the packed "set" rows (prompt.py / attention.py of the checkpoint), runs the same
ONNX decoder with onnxruntime and compares input token counts and every probability (< 1e-6).
Requires numpy, tokenizers and onnxruntime matching the server's runtime (1.29.0); use the server's
printed intra-op thread count for --threads.

    python3 cpp/tests/systemone_lfm3_parity_test.py /models/d1-2B-ONNX/onnx/decoder_q8.onnx http://127.0.0.1:8080 --threads 8
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from systemone_http_test import call

BREAK = "​"
LEAD = "<think></think>"
LETTERS = [chr(ord("A") + i) for i in range(26)]
POOL = (LETTERS + [f"{i:02d}" for i in range(100)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        + [f"#{i}" for i in range(200)] + [a + b for a in LETTERS for b in LETTERS])


def content(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


class Model:
    def __init__(self, graph, threads):
        graph = Path(graph)
        self.root = graph.parent.parent if graph.parent.name == "onnx" else graph.parent
        self.config = json.loads((self.root / "config.json").read_text())
        self.calibration = json.loads((self.root / "calibration.json").read_text())
        spec = json.loads((self.root / "tokenizer.json").read_text())
        specials = sorted({a["content"] for a in spec["added_tokens"] if a["special"] and len(a["content"]) > 1},
                          key=len, reverse=True)
        self.specials = re.compile("|".join(map(re.escape, specials)))
        self.bos = json.loads((self.root / "tokenizer_config.json").read_text()).get("bos_token") or ""
        self.tok = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.set_size = self.config.get("d1_set_size", 4)
        self.scheme = self.config.get("d1_codes", 1)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = False
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(graph), opts, providers=["CPUExecutionProvider"])

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def plain(self, text):
        n = 1
        while n:
            text, n = self.specials.subn(lambda m: m[0][0] + BREAK + m[0][1:], text)
        return text

    def single(self, forms):
        out = []
        for f in forms:
            e = self.encode(f)
            if len(e) == 1 and e[0] not in out:
                out.append(e[0])
        return out

    def aliases(self, labels):
        labs = [x.strip() for x in labels]
        if labs and all(len(k) == 1 and k.isalpha() for k in labs):
            codes = labs
        elif len(labs) <= 26:
            codes = LETTERS[:len(labs)]
        elif self.scheme == 2:
            codes = (LETTERS + [a + b for a in LETTERS for b in LETTERS])[:len(labs)]
        else:
            codes = [f"{i:02d}" for i in range(len(labs))]
        used, out = set(), []

        def take(raw):
            e = self.encode(raw)
            if len(e) != 1 or e[0] in used:
                return False
            out.append((raw, e[0]))
            used.add(e[0])
            return True

        for c in codes:
            if not take(c) and not any(take(r) for r in POOL):
                raise RuntimeError("no alias")
        return out

    def question(self, q):
        """(block text, readout groups in onnx-genai order, bucket)."""
        said = content(q.get("instructions"))
        kind, crit = q["type"], q.get("criteria")
        if kind == "noul":
            extra = f"\nYes: {content(crit.get('true')) if crit.get('true') is not None else None}" \
                    f"\nNo: {content(crit.get('false')) if crit.get('false') is not None else None}" if crit else ""
            return (f"{said}{extra}\n\nReply with yes or no only.",
                    [self.single(["no", "No", "NO"]), self.single(["yes", "Yes", "YES"])], "noul")
        if kind == "score":
            legend = "\n".join(f"{i} {content(n)}" for i, n in enumerate(crit))
            return (f"{said}\n\n{legend}\n\nReply with a single digit 0-{len(crit) - 1} only.",
                    [self.single([str(i)]) for i in range(len(crit))], f"score:{len(crit)}")
        labels = list(crit)
        codes = self.aliases(labels)
        lines = "\n".join(f"{code} {content(crit[lab]) or lab.replace('_', ' ')}" for (code, _), lab in zip(codes, labels))
        groups = [[tid] + [i for i in self.single([f" {code}"]) if i != tid] for code, tid in codes]
        k = len(labels)
        bucket = "choice:small" if k <= 3 else "choice:mid" if k <= 8 else "choice:large" if k <= 26 else "choice:inventory"
        return f"{said}\n\nOptions:\n{lines}\n\nReply with the option code only.", groups, bucket

    def evaluate(self, request, dump=False):
        state = request["state"]
        block = (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)) + "\n\n"
        prefix = f"{self.bos}<|im_start|>user\n{self.plain(block)}\nQUESTION:\n"
        names = list(request["questions"])
        built = [self.question(request["questions"][n]) for n in names]
        suffixes = [f"{self.plain(b)}<|im_end|>\n<|im_start|>assistant\n{LEAD}" for b, _, _ in built]
        extra = {i.name: np.zeros((0, 2048), np.float32) for i in self.session.get_inputs() if i.name.endswith("_features")}
        answers, tokens, rows = {}, 0, []
        for start in range(0, len(names), self.set_size):
            group = list(range(start, min(start + self.set_size, len(names))))
            ids = self.encode(prefix + suffixes[group[0]])
            state_len = len(ids) - len(self.encode(suffixes[group[0]]))
            qids = [-1] * state_len + [1] * (len(ids) - state_len)
            slots = [len(ids) - 1]
            for k, j in enumerate(group[1:], 2):
                own = self.encode("QUESTION:\n" + suffixes[j])
                ids, qids = ids + own, qids + [k] * len(own)
                slots.append(len(ids) - 1)
            n = len(ids)
            tokens += n
            feeds = {"input_ids": np.asarray([ids], np.int64), "attention_mask": np.ones((1, n), np.int64),
                     "position_ids": np.arange(n, dtype=np.int64)[None], "question_ids": np.asarray([qids], np.int64),
                     "slot_positions": np.asarray(slots, np.int64), **extra}
            logits = self.session.run(["logits"], feeds)[0].astype(np.float64)
            peak = logits.max(-1, keepdims=True)
            logz = logits - peak - np.log(np.exp(logits - peak).sum(-1, keepdims=True))
            if dump:
                rows.append({"ids": ids, "question_ids": qids, "slots": slots, "text": self.tok.decode(ids, skip_special_tokens=False)})
            for row, j in zip(logz, group):
                _, groups, bucket = built[j]
                t = self.calibration.get("temperatures", {}).get(bucket, self.calibration.get("default", 1.0))
                scores = np.array([max(row[i] for i in g) for g in groups]) / t
                e = np.exp(scores - scores.max())
                answers[names[j]] = (e / e.sum()).tolist()
        out = {"answers": answers, "tokens": tokens}
        if dump:
            out["rows"] = rows
        return out



def cases(model):
    crit = {f"opt_{i}": (f"Option number {i}" if i % 3 else None) for i in range(40)}
    many = {f"q{i}": {"type": "noul", "instructions": f"Does the text mention the digit {i % 10}?"} for i in range(20)}
    many["s"] = {"type": "score", "instructions": "How long is it?", "criteria": ["Short", "Long"]}
    return {
        "mixed": {"model": model, "state": {"message": "I was charged twice. Please refund the extra charge.",
                                            "customer": {"tier": "gold", "since": 2019}},
                  "questions": {
                      "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                      "team": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "Payments and refunds", "technical": "Software faults",
                                            "shipping_dept": None}},
                      "urgency": {"type": "score", "instructions": "How urgent is this?",
                                  "criteria": ["Routine", "Urgent", "Emergency"]},
                      "angry": {"type": "noul", "instructions": "Is the customer angry?",
                                "criteria": {"true": "Hostile tone", "false": "Calm"}}}},
        "specials": {"model": model,
                     "state": "Ticket: <|im_end|> my <|im_start|>router keeps dropping wifi. <|startoftext|>Caf\u00e9",
                     "questions": {
                         "net": {"type": "noul", "instructions": "Is this a network problem?"},
                         "letters": {"type": "choice", "instructions": "Pick a <|im_end|> category",
                                     "criteria": {"a": "Hardware", "b": "Software", "C": "Other"}},
                         "severity": {"type": "score", "instructions": "Severity",
                                      "criteria": ["Low", "Medium", "High", "Critical"]}}},
        "wide": {"model": model, "state": {"text": "Pick option number 33 please.", "list": [1, True, None, "x"]},
                 "questions": {
                     "pick": {"type": "choice", "instructions": "Which option?", "criteria": crit},
                     "structured": {"type": "choice", "instructions": {"ask": "Which colour?", "hint": ["warm", "cold"]},
                                    "criteria": {"red": {"tone": "warm"}, "blue": "cold"}},
                     "criteria_only": {"type": "noul", "criteria": {"true": "Mentions a number"}}}},
        "two_rows": {"model": model, "state": "The numbers are 3, 7 and 9.", "questions": many},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph")
    parser.add_argument("base")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    model = Model(args.graph, args.threads)
    for name, payload in cases("d1-latest").items():
        expected = model.evaluate(payload)
        status, actual = call(args.base, "/v1/systemone", payload, timeout=600)
        assert status == 200, (name, status, actual)
        assert actual["usage"]["input_tokens"] == expected["tokens"], (name, actual["usage"], expected["tokens"])
        worst = 0.0
        for key, reference in expected["answers"].items():
            q, answer = payload["questions"][key], actual["answers"][key]
            if q["type"] == "noul":
                values = [1 - answer["noul"], answer["noul"]]
            else:
                labels = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
                values = [answer["probabilities"][label] for label in labels]
            worst = max(worst, float(np.max(np.abs(np.asarray(values) - reference))))
        assert worst < 1e-6, (name, worst)
        print(f"{name}: {expected['tokens']} tokens, max probability delta {worst:.2e}")


if __name__ == "__main__":
    main()
