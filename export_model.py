import torch
import onnx
from transformers import AutoTokenizer, AutoModel
from onnxruntime_extensions import gen_processing_models
import onnx.compose

# Settings
model_name = "intfloat/e5-small-v2"
base_model_path = "base_model.onnx"
final_model_path = "fused_model.onnx"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# --- STEP 1: Export Base Model ---
print("Exporting base model...")
dummy_input = tokenizer("Test input", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    base_model_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"}
    },
    opset_version=17
)

# --- STEP 2: Generate Tokenizer ONNX ---
print("Generating tokenizer model...")
pre_model = gen_processing_models(
    tokenizer,
    pre_kwargs={"opset": 17}
)[0]

# --- STEP 3: Fix Version Mismatch & Merge ---
print("Loading base model for merging...")
base_onnx = onnx.load(base_model_path)
tokenizer_onnx = pre_model

# === THE FIX ===
# Check versions
print(f"Tokenizer IR Version: {tokenizer_onnx.ir_version}")
print(f"Base Model IR Version: {base_onnx.ir_version}")

if base_onnx.ir_version != tokenizer_onnx.ir_version:
    print(f"Fixing mismatch: Downgrading Base Model IR version from {base_onnx.ir_version} to {tokenizer_onnx.ir_version}")
    # We force the Base Model to match the Tokenizer (usually version 8)
    # This is safer than upgrading the tokenizer.
    base_onnx.ir_version = tokenizer_onnx.ir_version

# Merge
print("Merging models...")
combined_model = onnx.compose.merge_models(
    tokenizer_onnx,
    base_onnx,
    io_map=[
        ("input_ids", "input_ids"),
        ("attention_mask", "attention_mask")
    ]
)

onnx.save(combined_model, final_model_path)
print(f"Success! Fused model saved to {final_model_path}")
