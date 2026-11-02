"""Generate tiny, input-sensitive Laya ONNX/tokenizer fixtures; no weights/downloads."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors


def graph():
    inputs = [
        h.make_tensor_value_info("input_ids", T.INT64, ["B", "L"]),
        h.make_tensor_value_info("attention_mask", T.INT64, ["B", "L"]),
        h.make_tensor_value_info("marker_pos", T.INT64, ["B", "K"]),
        h.make_tensor_value_info("marker_mask", T.BOOL, ["B", "K"]),
        h.make_tensor_value_info("qtype", T.INT64, ["B"]),
    ]
    outputs = [
        h.make_tensor_value_info("logits", T.FLOAT, ["B", "K"]),
        h.make_tensor_value_info("act_probs", T.FLOAT, ["B", 2]),
    ]
    constants = {
        "one_i": np.array(1, np.int64), "axis": np.array([1], np.int64),
        "w_token": np.array(0.07, np.float32), "w_pos": np.array(0.013, np.float32),
        "w_sum": np.array(0.0001, np.float32), "one": np.array(1, np.float32),
        "negative": np.array(-10000, np.float32),
    }
    nodes = [
        h.make_node("Add", ["marker_pos", "one_i"], ["next_pos"]),
        h.make_node("GatherElements", ["input_ids", "next_pos"], ["option_id"], axis=1),
        h.make_node("Cast", ["option_id"], ["option_f"], to=T.FLOAT),
        h.make_node("Cast", ["marker_pos"], ["position_f"], to=T.FLOAT),
        h.make_node("Cast", ["qtype"], ["type_f"], to=T.FLOAT),
        h.make_node("Add", ["type_f", "one"], ["type_factor"]),
        h.make_node("Unsqueeze", ["type_factor", "axis"], ["type_column"]),
        h.make_node("Mul", ["option_f", "type_column"], ["option_type"]),
        h.make_node("Mul", ["option_type", "w_token"], ["token_score"]),
        h.make_node("Mul", ["position_f", "w_pos"], ["position_score"]),
        h.make_node("Mul", ["input_ids", "attention_mask"], ["masked_ids"]),
        h.make_node("ReduceSum", ["masked_ids", "axis"], ["sum_ids"], keepdims=1),
        h.make_node("Cast", ["sum_ids"], ["sum_f"], to=T.FLOAT),
        h.make_node("Mul", ["sum_f", "w_sum"], ["sum_scale"]),
        h.make_node("Mul", ["sum_scale", "position_f"], ["sum_score"]),
        h.make_node("Add", ["token_score", "position_score"], ["local_score"]),
        h.make_node("Add", ["local_score", "sum_score"], ["scores"]),
        h.make_node("Where", ["marker_mask", "scores", "negative"], ["logits"]),
        h.make_node("Sigmoid", ["type_f"], ["prob"]),
        h.make_node("Sub", ["one", "prob"], ["complement"]),
        h.make_node("Unsqueeze", ["prob", "axis"], ["prob_column"]),
        h.make_node("Unsqueeze", ["complement", "axis"], ["complement_column"]),
        h.make_node("Concat", ["complement_column", "prob_column"], ["act_probs"], axis=1),
    ]
    result = h.make_model(
        h.make_graph(nodes, "input_sensitive_laya", inputs, outputs, [
            numpy_helper.from_array(value, name) for name, value in constants.items()
        ]), opset_imports=[h.make_opsetid("", 17)], ir_version=8,
    )
    onnx.checker.check_model(result)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def legacy_fixtures(root):
    words = [
        "[UNK]", "no", "No", "NO", "yes", "Yes", "YES", "A", "B", "C", "0", "1", "2",
        "state", "alpha", "beta", "one", "low", "middle", "high", "choose", "rate",
        "<|state|>", "<|question|>", "<|option_start|>", "<|option_end|>", "<|decide|>",
    ]
    vocab = dict(zip(words, range(len(words))))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(words[-5:])
    kev = root / "dispatch-kev"
    d1 = root / "dispatch-d1"
    for target in (kev, d1):
        target.mkdir(exist_ok=True)
        tokenizer.save(str(target / "tokenizer.json"))
    delimiters = dict(zip(("state", "question", "option_start", "option_end", "decide"), words[-5:]))
    write_json(kev / "config.json", {"kev": {
        "delimiters": delimiters, "delimiter_ids": {key: vocab[value] for key, value in delimiters.items()},
        "max_state_tokens": 1024, "max_branch_tokens": 2048, "max_options": 255,
    }})
    write_json(d1 / "config.json", {"d1": {}, "text_config": {"max_position_embeddings": 2048}})

    def save(target, name, inputs, outputs, nodes, constants):
        model = h.make_model(h.make_graph(nodes, name, inputs, outputs, [
            numpy_helper.from_array(np.asarray(value), key) for key, value in constants.items()
        ]), opset_imports=[h.make_opsetid("", 17)], ir_version=8)
        onnx.checker.check_model(model)
        onnx.save(model, target / (name + ".onnx"))

    # Read the token immediately before each end marker, independent of packed offsets.
    save(kev, "model", [
        h.make_tensor_value_info("input_ids", T.INT64, [1, "L"]),
        h.make_tensor_value_info("attention_mask", T.INT64, [1, "L"]),
    ], [h.make_tensor_value_info("logits", T.FLOAT, [1, "L"])], [
        h.make_node("Pad", ["input_ids", "pads"], ["padded"]),
        h.make_node("Slice", ["padded", "start", "end", "axis"], ["previous"]),
        h.make_node("Cast", ["previous"], ["previous_f"], to=T.FLOAT),
        h.make_node("Cast", ["attention_mask"], ["mask_f"], to=T.FLOAT),
        h.make_node("Mul", ["previous_f", "weight"], ["scores"]),
        h.make_node("Mul", ["mask_f", "mask_weight"], ["mask_score"]),
        h.make_node("Add", ["scores", "mask_score"], ["logits"]),
    ], {"pads": np.array([0, 1, 0, 0], np.int64), "start": np.array([0], np.int64),
        "end": np.array([-1], np.int64), "axis": np.array([1], np.int64),
        "weight": np.array(0.1, np.float32), "mask_weight": np.array(0.001, np.float32)})
    save(d1, "embed_tokens", [h.make_tensor_value_info("input_ids", T.INT64, [1, "L"])],
         [h.make_tensor_value_info("embeddings", T.FLOAT, [1, "L", 1])], [
             h.make_node("Cast", ["input_ids"], ["ids_f"], to=T.FLOAT),
             h.make_node("Unsqueeze", ["ids_f", "axis"], ["embeddings"]),
         ], {"axis": np.array([2], np.int64)})
    # The existing d1 adapter reads final-position logits [1,1,V], not all L positions.
    save(d1, "decoder_model_merged", [
        h.make_tensor_value_info("inputs_embeds", T.FLOAT, [1, "L", 1]),
        h.make_tensor_value_info("attention_mask", T.INT64, [1, "L"]),
    ], [h.make_tensor_value_info("logits", T.FLOAT, [1, 1, len(words)])], [
        h.make_node("Cast", ["attention_mask"], ["mask_f"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["mask_f", "hidden_axis"], ["mask_3d"]),
        h.make_node("Mul", ["inputs_embeds", "mask_3d"], ["masked"]),
        h.make_node("ReduceSum", ["masked", "sequence_axis"], ["sum_ids"], keepdims=1),
        h.make_node("Mul", ["sum_ids", "sum_weight"], ["scaled_sum"]),
        h.make_node("Add", ["scaled_sum", "bias"], ["scale"]),
        h.make_node("Mul", ["scale", "vocabulary"], ["logits"]),
    ], {"hidden_axis": np.array([2], np.int64), "sequence_axis": np.array([1], np.int64),
        "sum_weight": np.array(0.0001, np.float32), "bias": np.array(0.07, np.float32),
        "vocabulary": np.arange(len(words), dtype=np.float32).reshape(1, 1, -1)})


def generate(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    words = [
        "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]",
        "choice", "score", "noul", "question", ":", "choose", "rate", "true",
        "state", "alpha", "beta", "one", "low", "middle", "high", "level",
        "0", "1", "2", "false", "no", ",", "the", "statement", "does", "not",
        "hold", "yes", "holds", "?", "é", "second", "longer", "instructions",
    ]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(words[:5])
    # Native code must disable automatic special tokens and tokenizer truncation/padding.
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 0), ("[SEP]", 1)]
    )
    tokenizer.enable_truncation(12)
    tokenizer.enable_padding(length=16, pad_id=3, pad_token="[PAD]")
    (root / "tokenizer").mkdir(exist_ok=True)
    tokenizer.save(str(root / "tokenizer" / "tokenizer.json"))
    special = dict(zip(("cls_token", "sep_token", "mask_token", "pad_token"), words[:4]))
    write_json(root / "tokenizer" / "tokenizer_config.json", special)
    metadata = {
        "max_len": 1024, "head_max_len": 128, "temperature": [2, 3, 4],
        "temperature_by_options": {"choice:2": 0.5, "score:3-5": 1.5, "noul:2": 0.75},
    }
    write_json(root / "laya_config.json", metadata)
    base = graph()
    onnx.save(base, root / "model.onnx")

    def clone(name, rejected=True):
        target = root / ("reject" if rejected else "") / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / "tokenizer", target / "tokenizer", dirs_exist_ok=True)
        shutil.copyfile(root / "laya_config.json", target / "laya_config.json")
        shutil.copyfile(root / "model.onnx", target / "model.onnx")
        return target

    nested = clone("nested-specials", False)
    write_json(nested / "tokenizer" / "tokenizer_config.json", {
        key: {"content": value, "special": True} for key, value in special.items()
    })
    fallback = clone("no-buckets", False)
    write_json(fallback / "laya_config.json", {key: value for key, value in metadata.items()
                                             if key != "temperature_by_options"})
    for backend in ("kev", "d1"):
        write_json(clone("ambiguous-" + backend) / "config.json", {backend: {}})
    (clone("missing-metadata") / "laya_config.json").unlink()
    (clone("malformed-metadata") / "laya_config.json").write_text("{")
    write_json(clone("metadata-array") / "laya_config.json", [])
    for key in ("temperature", "head_max_len", "max_len"):
        bad = deepcopy(metadata)
        del bad[key]
        write_json(clone("missing-" + key) / "laya_config.json", bad)
    for key in special:
        bad = dict(special)
        bad[key] = {"content": "[UNK]"}
        write_json(clone("wrong-" + key) / "tokenizer" / "tokenizer_config.json", bad)
    bad = dict(special)
    del bad["cls_token"]
    write_json(clone("missing-special") / "tokenizer" / "tokenizer_config.json", bad)
    unregistered = json.loads((root / "tokenizer" / "tokenizer.json").read_text())
    del unregistered["model"]["vocab"]["[MASK]"]
    unregistered["added_tokens"] = [token for token in unregistered["added_tokens"]
                                   if token["content"] != "[MASK]"]
    write_json(clone("missing-mask-vocab") / "tokenizer" / "tokenizer.json", unregistered)

    def variant(name, change):
        model = deepcopy(base)
        change(model)
        onnx.save(model, clone(name) / "model.onnx")

    variant("static-length", lambda m: setattr(m.graph.input[0].type.tensor_type.shape.dim[1], "dim_value", 32))
    variant("static-batch", lambda m: setattr(m.graph.input[1].type.tensor_type.shape.dim[0], "dim_value", 1))
    variant("static-marker-width", lambda m: setattr(m.graph.input[2].type.tensor_type.shape.dim[1], "dim_value", 3))
    variant("wrong-qtype-rank", lambda m: m.graph.input[4].type.tensor_type.shape.dim.add().CopyFrom(
        h.make_tensor_value_info("unused", T.INT64, ["extra"]).type.tensor_type.shape.dim[0]))
    variant("wrong-ids-type", lambda m: setattr(m.graph.input[0].type.tensor_type, "elem_type", T.INT32))
    variant("wrong-attention-type", lambda m: setattr(m.graph.input[1].type.tensor_type, "elem_type", T.FLOAT))
    variant("wrong-marker-pos-type", lambda m: setattr(m.graph.input[2].type.tensor_type, "elem_type", T.INT32))
    variant("wrong-marker-mask-type", lambda m: setattr(m.graph.input[3].type.tensor_type, "elem_type", T.INT64))
    variant("wrong-qtype-type", lambda m: setattr(m.graph.input[4].type.tensor_type, "elem_type", T.INT32))
    variant("missing-input", lambda m: m.graph.input.pop())
    variant("extra-input", lambda m: m.graph.input.append(h.make_tensor_value_info("extra", T.INT64, ["B"])))
    variant("wrong-logits-type", lambda m: setattr(m.graph.output[0].type.tensor_type, "elem_type", T.DOUBLE))
    variant("wrong-act-type", lambda m: setattr(m.graph.output[1].type.tensor_type, "elem_type", T.DOUBLE))
    variant("wrong-act-width", lambda m: setattr(m.graph.output[1].type.tensor_type.shape.dim[1], "dim_value", 3))
    variant("static-output", lambda m: setattr(m.graph.output[0].type.tensor_type.shape.dim[1], "dim_value", 3))
    variant("missing-output", lambda m: m.graph.output.pop())
    variant("wrong-output-rank", lambda m: m.graph.output[0].type.tensor_type.shape.dim.pop())
    for name, output, value, operation in (
        ("nan-logits", "logits", np.nan, "Add"),
        ("infinite-logits", "logits", np.inf, "Add"),
        ("nan-act", "act_probs", np.nan, "Add"),
        ("negative-act", "act_probs", -1, "Mul"),
        ("unnormalized-act", "act_probs", 0, "Mul"),
    ):
        model = deepcopy(base)
        for node in model.graph.node:
            for i, current in enumerate(node.output):
                if current == output:
                    node.output[i] = output + "_original"
        model.graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), "invalid"))
        model.graph.node.append(h.make_node(operation, [output + "_original", "invalid"], [output]))
        onnx.checker.check_model(model)
        target = clone("runtime-" + name, False)
        onnx.save(model, target / "model.onnx")
    legacy_fixtures(root)
    print(f"Laya fixtures generated in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", help="Fresh project-relative fixture directory")
    generate(parser.parse_args().destination)
