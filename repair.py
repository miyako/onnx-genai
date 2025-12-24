import sys
import torch
import onnx
import onnx.compose
from transformers import AutoTokenizer, AutoModel
from onnxruntime_extensions import gen_processing_models

# -----------------------------
# 1. Setup
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: python fix_clean_sweep.py <model_id> <output.onnx>")
    sys.exit(1)

model_id = sys.argv[1]
output_path = sys.argv[2]

print(f"🚀 Starting Clean Sweep Repair for: {model_id}")

# -----------------------------
# 2. Export Components
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

# Dummy input
dummy = tokenizer(
    ["test"],
    return_tensors="pt",
    padding=False,
    truncation=True,
    return_token_type_ids=False
)

print("   1. Exporting Encoder (Opset 17)...")
dynamic_axes = {
    "input_ids":       {0: "batch", 1: "seq_len"},
    "attention_mask":  {0: "batch", 1: "seq_len"},
    "last_hidden_state": {0: "batch", 1: "seq_len"}
}

# Export Encoder (without token_type_ids in signature)
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "encoder.onnx",
    input_names=["input_ids", "attention_mask"], # Note: token_type_ids is NOT here
    output_names=["last_hidden_state"],
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True
)

print("   2. Exporting Tokenizer (Opset 18)...")
tokenizer_onnx = gen_processing_models(
    tokenizer,
    pre_kwargs={
        "padding": False,
        "truncation": False,
        "return_tensors": "np",
        "return_token_type_ids": False
    },
    opset_version=18
)[0]
onnx.save(tokenizer_onnx, "tokenizer.onnx")

# -----------------------------
# 3. Merge
# -----------------------------
print("   3. Merging...")
tok = onnx.load("tokenizer.onnx")
enc = onnx.load("encoder.onnx")

max_ir = max(enc.ir_version, tok.ir_version)
enc.ir_version = tok.ir_version = max_ir
for op in enc.opset_import: op.version = 18
for op in tok.opset_import: op.version = 18

fused = onnx.compose.merge_models(
    tok,
    enc,
    io_map=[("input_ids", "input_ids"), ("attention_mask", "attention_mask")]
)

# -----------------------------
# 4. SURGICAL REMOVAL (The Kill List)
# -----------------------------
# We delete the Slices (crash 1) AND the Expands (crash 2)
# Since token_type_ids aren't used, deleting Expand is safe.
NODES_TO_KILL = ["node_slice_1", "node_slice_2", "node_expand", "node_expand_1"]

def remove_dead_nodes(node_list, context="Graph"):
    nodes_to_keep = []
    replacement_map = {}

    for node in node_list:
        # A. Recurse into Subgraphs
        for attr in node.attribute:
            if attr.type == 5: remove_dead_nodes(attr.g.node, f"{context}->Sub")
            elif attr.type == 9:
                for g in attr.graphs: remove_dead_nodes(g.node, f"{context}->Sub")

        # B. Check Kill List
        if node.name in NODES_TO_KILL:
            print(f"      [{context}] 🔪 Deleting '{node.name}' ({node.op_type})")
            
            # Map output to input (bypass) just in case something downstream checks for existence
            if len(node.input) > 0 and len(node.output) > 0:
                replacement_map[node.output[0]] = node.input[0]
            continue

        nodes_to_keep.append(node)

    # C. Apply Rewiring
    if replacement_map:
        for node in nodes_to_keep:
            for i, inp in enumerate(node.input):
                if inp in replacement_map:
                    node.input[i] = replacement_map[inp]

    # D. Commit changes
    del node_list[:]
    node_list.extend(nodes_to_keep)

print("   4. removing broken/unused nodes...")
remove_dead_nodes(fused.graph.node, "MainGraph")

# Also scan functions
if hasattr(fused, 'functions'):
    for func in fused.functions:
        remove_dead_nodes(func.node, f"Func:{func.name}")

onnx.save(fused, output_path)
print(f"✅ Clean Model Saved: {output_path}")
