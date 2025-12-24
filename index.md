---
layout: default
---

![version](https://img.shields.io/badge/version-20%2B-E23089)
![platform](https://img.shields.io/static/v1?label=platform&message=mac-intel%20|%20mac-arm%20|%20win-64&color=blue)
[![license](https://img.shields.io/github/license/miyako/onnx-genai)](LICENSE)
![downloads](https://img.shields.io/github/downloads/miyako/onnx-genai/total)

# Use ONNX Runtime from 4D

#### Abstract

[**ONNX** (Open Neural Network eXchange)](https://github.com/onnx/onnx) is an open-source standard to represent machine learning models. Unlike the GGUF format which requires an inference engine for every model architecture as well as hardware optimisation for every chip architecture, ONNX is **framework agnostic** since the file contains the **computation graph of the neural network**.

To use ONNX you must first convert a plain model to ONNX fornat.

#### Usage

Instantiate `cs.onnx.onnx` in your *On Startup* database method:

```4d

```

Unless the server is already running (in which case the costructor does nothing), the following procedure runs in the background:

1. The specified model is downloaded via HTTP
2. The `onnx-genai` program is started

Now you can test the server:

```
curl -X 'POST' \
  'http://127.0.0.1:8080/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain quantum computing in one sentence."
      }
    ],
    "temperature": 0.7
  }'
```

Or, use AI Kit:

```4d
var $ChatCompletionsParameters : cs.AIKit.OpenAIChatCompletionsParameters
$ChatCompletionsParameters:=cs.AIKit.OpenAIChatCompletionsParameters.new({model: ""})

$ChatCompletionsParameters.max_completion_tokens:=2048
$ChatCompletionsParameters.n:=1
$ChatCompletionsParameters.temperature:=0.7
//%W-550.26
$ChatCompletionsParameters.top_k:=50
$ChatCompletionsParameters.top_p:=0.9
//%W+550.26
$ChatCompletionsParameters.body:=Formula($0:={\
top_k: This.top_k; \
top_p: This.top_p; \
temperature: This.temperature; \
n: This.n; \
max_completion_tokens: This.max_completion_tokens})
$messages:=[]
$messages.push({role: "system"; content: "You are a helpful assistant."})
$messages.push({role: "user"; content: "The window was shattered. Inside the room were 3 cats, a piano, 1 million dollars, a baseball bat, a bar of soap. What happened?"})

var $OpenAI : cs.AIKit.OpenAI
$OpenAI:=cs.AIKit.OpenAI.new({baseURL: "http://127.0.0.1:8080/v1"})

var $ChatCompletionsResult : cs.AIKit.OpenAIChatCompletionsResult
$ChatCompletionsResult:=$OpenAI.chat.completions.create($messages; $ChatCompletionsParameters)
If ($ChatCompletionsResult.success)
    ALERT($ChatCompletionsResult.choice.message.text)
End if 
```

#### Chat Completions Model

Use [optimum-cli](https://github.com/huggingface/optimum) to convert a specific model to ONNX:

```sh
optimum-cli export onnx --model BAAI/bge-base-en-v1.5 onnx_output_dir/
```

Or, downloaded a converted ONNX model:

```
hf download microsoft/Phi-3.5-mini-instruct-onnx \
  --include "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*" \
  --local-dir .
```

#### Embeddings Model

The model needs to be customised to accept strings not integers.

> Use conda to avoid "python too new" issue with `python3 -m venv`.

```bash
# 1. Download the installer
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

# 2. Run the installer (Type 'yes' to license and initialization prompts)
bash Miniforge3-MacOSX-arm64.sh

# 3. Refresh your shell so conda commands work
source ~/.zshrc
```

```bash
# 1. Create a new environment named '.onnx_env' with Python 3.13
conda create -n .onnx_env python=3.10 -y

# 2. Activate the environment
conda activate .onnx_env

# 3. Install onnxruntime-extensions
pip install "torch>=2.0.0" "transformers>=4.30.0" "onnx>=1.14.0" "onnxruntime>=1.16.0" "onnxruntime-extensions>=0.8.0"
```

```python
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
```

Finally to terminate the server:

```4d
var $onnx : cs.onnx
$onnx:=cs.onnx.new()
$onnx.terminate()
```

#### AI Kit compatibility

The API is compatibile with [Open AI](https://platform.openai.com/docs/api-reference/embeddings). 

|Class|API|Availability|
|-|-|:-:|
|Models|`/v1/models`|✅|
|Chat|`/v1/chat/completions`|✅|
|Images|`/v1/images/generations`||
|Moderations|`/v1/moderations`||
|Embeddings|`/v1/embeddings`||
|Files|`/v1/files`||
