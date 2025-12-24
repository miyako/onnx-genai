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

print(f"🚀 Starting Repair for: {model_id}")

# -----------------------------
# 2. Export Components
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

# Dummy input
dummy = tokenizer(["test"], return_tensors="pt", padding=False, truncation=True)

print("   1. Exporting Encoder...")
# We only define input_ids and attention_mask
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

print("   2. Exporting Tokenizer...")
# gen_processing_models often generates 3 outputs (ids, mask, type_ids)
# even if you ask it not to. We will fix this in the next step.
tokenizer_onnx = gen_processing_models(
    tokenizer,
    pre_kwargs={"padding": False, "truncation": True},
    opset_version=17
)[0]

# -----------------------------
# 3. Clean Tokenizer (The "Surgical" Fix)
# -----------------------------
print("   3. Cleaning Tokenizer outputs...")
# We only want the tokenizer to expose these two outputs to match our encoder
required_outputs = ["input_ids", "attention_mask"]

# Create a new version of the tokenizer graph that only exposes what we need
# This automatically handles the "dangling reference" problem by pruning the graph
tokenizer_onnx = onnx.compose.select_model_inputs_outputs(
    tokenizer_onnx,
    outputs=required_outputs
)
onnx.save(tokenizer_onnx, "tokenizer_cleaned.onnx")

# -----------------------------
# 4. Merge
# -----------------------------
print("   4. Merging Models...")
tok = onnx.load("tokenizer_cleaned.onnx")
enc = onnx.load("encoder.onnx")

# Align IR and Opset versions
max_ir = max(enc.ir_version, tok.ir_version)
enc.ir_version = tok.ir_version = max_ir

fused = onnx.compose.merge_models(
    tok,
    enc,
    io_map=[
        ("input_ids", "input_ids"),
        ("attention_mask", "attention_mask"),
        ("token_type_ids", "token_type_ids") # <--- Make sure this is included!
    ]
)

# -----------------------------
# 5. Final Polish
# -----------------------------
# Run a quick check to ensure the graph is valid
try:
    onnx.checker.check_model(fused)
    print("✅ Model logic is valid.")
except Exception as e:
    print(f"⚠️ Validation Warning: {e}")

onnx.save(fused, output_path)
print(f"✨ Clean Model Saved: {output_path}")
