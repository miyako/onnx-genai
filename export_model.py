import sys
import torch
import onnx
import onnx.compose
import numpy as np
from onnx import helper, numpy_helper
from transformers import AutoTokenizer, AutoModel
from onnxruntime_extensions import gen_processing_models

# -----------------------------
# 1. Setup
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: python fix_static.py <model_id> <output.onnx>")
    sys.exit(1)

model_id = sys.argv[1]
output_path = sys.argv[2]

print(f"🚀 Starting Static Constant Repair for: {model_id}")

# -----------------------------
# 2. Export Components
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

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

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "encoder.onnx",
    input_names=["input_ids", "attention_mask"],
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
# Patch Encoder to look like Opset 18
for op in enc.opset_import: op.version = 18
for op in tok.opset_import: op.version = 18

fused = onnx.compose.merge_models(
    tok,
    enc,
    io_map=[("input_ids", "input_ids"), ("attention_mask", "attention_mask")]
)

# -----------------------------
# 4. STATIC DATA REPAIR
# -----------------------------
NODES_TO_KILL = ["node_slice_1", "node_slice_2"]

def repair_graph_static(graph):
    """
    1. Removes Slice nodes.
    2. Finds inputs to Expand nodes.
    3. If input is a Scalar Constant/Initializer -> Converts it to 1D Tensor in-place.
    """
    # Map Initializers for easy lookup/modification
    init_map = {init.name: init for init in graph.initializer}
    # Map Constant Nodes for easy lookup
    const_map = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}
    
    nodes_to_keep = []
    replacement_map = {}

    for node in graph.node:
        # A. Recurse
        for attr in node.attribute:
            if attr.type == 5: repair_graph_static(attr.g)
            elif attr.type == 9:
                for g in attr.graphs: repair_graph_static(g)

        # B. KILL SLICES
        if node.name in NODES_TO_KILL:
            print(f"      🔪 Deleting '{node.name}'")
            if len(node.input) > 0 and len(node.output) > 0:
                replacement_map[node.output[0]] = node.input[0]
            continue

        # C. FIX EXPAND (Static Update)
        if node.op_type == "Expand":
            if len(node.input) >= 2:
                shape_input_name = node.input[1]
                
                # Check 1: Is it an Initializer?
                if shape_input_name in init_map:
                    tensor = init_map[shape_input_name]
                    arr = numpy_helper.to_array(tensor)
                    if arr.ndim == 0:
                        print(f"      🔧 Fixing Initializer '{shape_input_name}' for Expand '{node.name}' (Scalar -> 1D)")
                        new_tensor = numpy_helper.from_array(np.array([arr.item()], dtype=arr.dtype), name=shape_input_name)
                        graph.initializer.remove(tensor)
                        graph.initializer.append(new_tensor)
                        init_map[shape_input_name] = new_tensor # Update map

                # Check 2: Is it a Constant Node?
                elif shape_input_name in const_map:
                    const_node = const_map[shape_input_name]
                    for attr in const_node.attribute:
                        if attr.name == "value":
                            tensor = attr.t
                            arr = numpy_helper.to_array(tensor)
                            if arr.ndim == 0:
                                print(f"      🔧 Fixing Constant Node '{const_node.name}' for Expand '{node.name}' (Scalar -> 1D)")
                                new_tensor = numpy_helper.from_array(np.array([arr.item()], dtype=arr.dtype))
                                attr.t.CopyFrom(new_tensor)
                
                # Check 3: Is it Dynamic? (Fallback to Reshape injection)
                else:
                    # If it's not a const/init, it's a dynamic output. We inject Reshape[-1].
                    # Only do this if we couldn't fix it statically.
                    # (Usually the tokenizer Expand nodes use constants, so the above catches 99% of cases)
                    print(f"      ⚠️ Dynamic input detected for Expand '{node.name}'. Injecting Reshape...")
                    import uuid
                    from onnx import TensorProto
                    
                    # Create helper const if missing
                    flat_name = "global_flat_const_static_fix"
                    if not any(i.name == flat_name for i in graph.initializer):
                        graph.initializer.append(helper.make_tensor(flat_name, TensorProto.INT64, [1], [-1]))
                    
                    unique = uuid.uuid4().hex[:6]
                    reshape_out = f"{shape_input_name}_dynamic_fix_{unique}"
                    reshape_node = helper.make_node(
                        "Reshape",
                        inputs=[shape_input_name, flat_name],
                        outputs=[reshape_out],
                        name=f"Reshape_Dynamic_{unique}"
                    )
                    nodes_to_keep.append(reshape_node)
                    node.input[1] = reshape_out

        nodes_to_keep.append(node)

    # D. Rewire Deleted Nodes
    if replacement_map:
        for output in graph.output:
            if output.name in replacement_map: output.name = replacement_map[output.name]
        for node in nodes_to_keep:
            for i, inp in enumerate(node.input):
                if inp in replacement_map: node.input[i] = replacement_map[inp]

    del graph.node[:]
    graph.node.extend(nodes_to_keep)

# -----------------------------
# 5. Execute
# -----------------------------
print("   4. Running Static Repair...")
repair_graph_static(fused.graph)

# Handle functions too
if hasattr(fused, 'functions'):
    for func in fused.functions:
        # Pass main graph initializer for global context if needed
        repair_graph_static(func)

onnx.save(fused, output_path)
print(f"✅ Final Static-Fixed Model Saved: {output_path}")
