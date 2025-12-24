import sys
import torch
import onnx
import onnx.compose
from transformers import AutoTokenizer, AutoModel
from onnxruntime_extensions import gen_processing_models

# Check if arguments are provided
if len(sys.argv) < 3:
    print("Usage: python export_model.py <input_model.onnx> <output_model.onnx>")
    sys.exit(1)

# Configuration
model_id = sys.argv[1]        # 1st argument: Existing ONNX model
output_file = sys.argv[2]       # 2nd argument: Output filename

print(f"Loading {model_id}...")

# 1. Define the Core Model (Transformer + Mean Pooling)
# We must wrap this because standard HF models usually output raw hidden states, not embeddings.
class CoreModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Run BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Mean Pooling (Standard for Sentence-Transformers)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings

# Initialize and set to eval mode
core_model = CoreModel(model_id)
core_model.eval()

# 2. Export the Core Model to ONNX
print("Exporting Core Model...")
dummy_input = "This is a test sentence."
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(dummy_input, return_tensors="pt")

# Note: We explicitly name inputs to match what the Tokenizer ONNX will output later
torch.onnx.export(
    core_model,
    (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
    "core.onnx",
    opset_version=17,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"}
    }
)

# 3. Generate the Tokenizer ONNX Model
print("Generating Tokenizer Graph...")
# gen_processing_models returns a tuple (pre_processor, post_processor). We only need 'pre'.
tokenizer_onnx = gen_processing_models(tokenizer, pre_kwargs={})[0]
onnx.save_model(tokenizer_onnx, "tokenizer.onnx")

# 4. Merge Tokenizer + Core Model
print("Merging models...")
core_model_proto = onnx.load_model("core.onnx")
tokenizer_model_proto = onnx.load_model("tokenizer.onnx")

# --- FIX START: Align Versions ---
# 1. Align IR Version (Intermediate Representation)
# We set both to the higher version to satisfy the strict equality check
common_ir_version = max(core_model_proto.ir_version, tokenizer_model_proto.ir_version)
core_model_proto.ir_version = common_ir_version
tokenizer_model_proto.ir_version = common_ir_version

# 2. Align Opset Version (Operator Set)
# merge_models also requires the default "ai.onnx" opset versions to match.
# We find the default opset in both models and update the lower one to match the higher one.
def get_default_opset(model):
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            return opset
    return None

core_opset = get_default_opset(core_model_proto)
tokenizer_opset = get_default_opset(tokenizer_model_proto)

if core_opset and tokenizer_opset:
    common_opset_version = max(core_opset.version, tokenizer_opset.version)
    print(f"Aligning Opset versions to: {common_opset_version}")
    core_opset.version = common_opset_version
    tokenizer_opset.version = common_opset_version
# --- FIX END ---

# This function fuses the two models by connecting the outputs of 'tokenizer' to inputs of 'core'
combined_model = onnx.compose.merge_models(
    tokenizer_model_proto,
    core_model_proto,
    io_map=[
        ("input_ids", "input_ids"),
        ("attention_mask", "attention_mask"),
        ("token_type_ids", "token_type_ids")
    ]
)

onnx.save_model(combined_model, output_file)
print(f"Success! Model saved to {output_file}")
