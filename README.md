# onnx-genai

ONNX Runtime GenAI Inference Engine

```
Usage:  onnx-genai -s -m chat_completion_model -e embedding_model -p port 

 -m path     : chat completion model
 -e path     : embedding model (pooling=mean)
 -r path     : rerank model
 -j          : read chat template from stdin
 -t path     : read chat template from path
 -d          : pooling=e2e (Universal Sentence Encoder)
 -b          : pooling=multi-vector (ColBERT)
 -l          : pooling=last-token (Llama)
 -c          : pooling=cls
 -s          : server (OpenAI compatible endpoint)
 -p          : server listening port (default=8080)
 -h host     : server host (default=127.0.0.1)    
```

## Dependencies

- `onnxruntime-genai-0.12.0`
- `onnxruntime-extensions-0.14.0`
- `onnxruntime-1.25.0`

## OpenAI Compatible Endpoints

- `/v1/models`
- `/v1/chat/completions`
- `/v1/embeddings`

## Cohere Compatible Endpoints

- `/v1/rerank`

## Converted ONNX Models

### Chat Completion

||`int4`|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`
|-|-:|-:|-:|-:|
|[`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)|[`1860`](https://huggingface.co/keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4)|`8192`|`2048`|`16`|
|[`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct)|[`4860`](https://huggingface.co/keisuke-miyako/Phi-4-mini-instruct-onnx-int4)|`131072`|`3072`|`32`|
|[`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)|[`2720`](https://huggingface.co/keisuke-miyako/Phi-3.5-mini-instruct-onnx-int4)|`131072`|`3072`|`32`|
|[`microsoft/Phi-4-mini-reasoning`](https://huggingface.co/microsoft/Phi-4-mini-reasoning)|[`4860`](https://huggingface.co/keisuke-miyako/Phi-4-mini-reasoning-onnx-int4)|`131072`|`3072`|`32`|
|[`microsoft/Phi-4-reasoning`](https://huggingface.co/microsoft/Phi-4-reasoning)|[`1090`](https://huggingface.co/keisuke-miyako/Phi-4-reasoning-onnx-int4)|`32768`|`5120`|`40`|
|[`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4)|[`1090`](https://huggingface.co/keisuke-miyako/Phi-4-onnx-int4)|`16384`|`5120`|`40`|
|[`microsoft/Phi-4-reasoning-plus`](https://huggingface.co/microsoft/Phi-4-reasoning-plus)|[`1090`](keisuke-miyako/Phi-4-reasoning-plus-onnx-int4)|`32768`|`5120`|`40`|
|[`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)|[`5380`](https://huggingface.co/keisuke-miyako/gemma-3-4b-it-onnx-int4)|`131072`|`2560`|`34`|
|[`google/translategemma-4b-it`](https://huggingface.co/google/translategemma-4b-it)|[`5380`](https://huggingface.co/keisuke-miyako/translategemma-4b-it-onnx-int4)|`131072`|`2560`|`34`|
|[`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it)|[`1900`](https://huggingface.co/keisuke-miyako/gemma-3-1b-it-onnx-int4)|`32768`|`1152`|`26`|
|[`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it)|[`906`](https://huggingface.co/keisuke-miyako/gemma-3-270m-it-onnx-int4)|`32768`|`640`|`18`|
|[`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it)|[`906`](https://huggingface.co/keisuke-miyako/functiongemma-270m-it-onnx-int4)|`32768`|`640`|`18`|
|[`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)|[`4010`](https://huggingface.co/keisuke-miyako/gemma-2-2B-it-onnx-int4)|`8192`|`2304`|`26`|
|[`google/gemma-2-2b-jpn-it`](https://huggingface.co/google/gemma-2-2b-jpn-it)|[`4010`](https://huggingface.co/keisuke-miyako/gemma-2-2b-jpn-it-onnx-int4)|`8192`|`2304`|`26`|
|[`OpenLLM-France/Lucie-7B-Instruct-v1.1`](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1)|[`5110`](https://huggingface.co/keisuke-miyako/Lucie-7B-Instruct-v1.1-onnx-int4)|`32000`|`4096`|`32`|
|[`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)|[`1470`](https://huggingface.co/keisuke-miyako/SmolLM2-1.7B-onnx-int4)|`8192`|`2048`|`24`|
|[`ibm-granite/granite-3.0-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct)|[`1990`](https://huggingface.co/keisuke-miyako/granite-3.0-2b-instruct-onnx-int4)|`4096`|`2048`|`40`|
|[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)|[`2020`](https://huggingface.co/keisuke-miyako/granite-3.3-2b-instruct-onnx-int4)|`131072`|`2048`|`40`|
|[`utter-project/EuroLLM-1.7B-Instruct`](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct)|[`1920`](https://huggingface.co/keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4)|`4096`|`2048`|`24`|
|[`utter-project/EuroLLM-9B-Instruct`](https://huggingface.co/utter-project/EuroLLM-9B-Instruct)|[`7490`](https://huggingface.co/keisuke-miyako/EuroLLM-9B-Instruct-onnx-int4)|`4096`|`4096`|`42`|
|[`h2oai/h2o-danube-1.8b-chat`](https://huggingface.co/h2oai/h2o-danube-1.8b-chat)|[`1430`](https://huggingface.co/keisuke-miyako/h2o-danube-1.8b-chat-onnx-int4)|`16384`|`2560`|`24`|
|[`h2oai/h2o-danube2-1.8b-chat`](https://huggingface.co/h2oai/h2o-danube2-1.8b-chat)|[`1430`](https://huggingface.co/keisuke-miyako/h2o-danube2-1.8b-chat-onnx-int4)|`8192`|`2560`|`24`|
|[`PleIAs/Baguettotron`](https://huggingface.co/PleIAs/Baguettotron)|[`353`](https://huggingface.co/keisuke-miyako/Baguettotron-onnx-int4)|`4096`|`576`|`80`|
|[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)|[`2340`](https://huggingface.co/keisuke-miyako/Qwen3-1.7B-onnx-int4)|`40960`|`2048`|`28`|
|[`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)|[`4210`](keisuke-miyako/Qwen3-4B-Thinking-2507-onnx-int4)|`262144`|`2560`|`36`|
|[`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B)|[`1920`](https://huggingface.co/keisuke-miyako/Qwen2.5-1.5B-onnx-int4)|`131072`|`1536`|`28`|
|[`deepseek-ai/deepseek-coder-1.3b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)|[`1080`](https://huggingface.co/keisuke-miyako/deepseek-coder-1.3b-instruct-onnx-int4)|`16384`|`2048`|`24`|
|[`01-ai/Yi-Coder-1.5B-Chat`](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat)|[`1440`](https://huggingface.co/keisuke-miyako/Yi-Coder-1.5B-Chat-onnx-int4)|`131072`|`2048`|`24`|
|[`OpenLLM-France/Claire-7B-FR-Instruct-0.1`](https://huggingface.co/OpenLLM-France/Claire-7B-FR-Instruct-0.1)|[`6920`](https://huggingface.co/keisuke-miyako/Claire-7B-FR-Instruct-0.1-onnx-int4)|`2048`|`4544`|`32`|

### Rerank

||`fp16`|`fp32`|`int8`|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`
|-|-:|-:|-:|-:|-:|-:
|[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)|[`45`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp16)|[`91`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp32)|[`23`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-int8)|`512`|`384`|`6`
|[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)|[`235`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp32)|[`118`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-int8)|`512`|`384`|`12`|
|[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)|[`1140`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp32)|[`569`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-int8)|`8192`|`1024`|`24`|
|[`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base)|[`556`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-int8)|`8192`|`768`|`12`|
|[`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large)|[`1120`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp32)|[`561`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-int8)|`8192`|`1024`|`24`
|[`jinaai/jina-reranker-v1-turbo-en`](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)|[`75`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp16)|[`151`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp32)|[`38`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-int8)|`8192`|`384`|`6`
|[`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2)|[`299`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp16)|[`599`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-int8)|`8192`|`768`|`22`
|[`hotchpotch/japanese-reranker-tiny-v2`](https://huggingface.co/hotchpotch/japanese-reranker-tiny-v2)|[`58`](https://huggingface.co/keisuke-miyako/japanese-reranker-tiny-v2-onnx-fp16)|[`117`](https://huggingface.co/keisuke-miyako/japanese-reranker-tiny-v2-onnx-fp32)|[`29`](https://huggingface.co/keisuke-miyako/japanese-reranker-tiny-v2-onnx-int8)|`8192`|`256`|`3`
|[`hotchpotch/japanese-reranker-xsmall-v2`](https://huggingface.co/hotchpotch/japanese-reranker-xsmall-v2)|[`73`](https://huggingface.co/keisuke-miyako/japanese-reranker-xsmall-v2-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/japanese-reranker-xsmall-v2-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/japanese-reranker-xsmall-v2-onnx-int8)|`8192`|`256`|`10`
|[`hotchpotch/japanese-reranker-small-v2`](https://huggingface.co/hotchpotch/japanese-reranker-small-v2)|[`140`](https://huggingface.co/keisuke-miyako/japanese-reranker-small-v2-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/japanese-reranker-small-v2-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/japanese-reranker-small-v2-onnx-int8)|`8192`|`384`|`13`
|[`hotchpotch/japanese-reranker-base-v2`](https://huggingface.co/hotchpotch/japanese-reranker-base-v2)|[`265`](https://huggingface.co/keisuke-miyako/japanese-reranker-base-v2-onnx-fp16)|[`530`](https://huggingface.co/keisuke-miyako/japanese-reranker-base-v2-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/japanese-reranker-base-v2-onnx-int8)|`8192`|`512`|`19`
|[`cl-nagoya/ruri-v3-reranker-310m`](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)|[`630`](https://huggingface.co/keisuke-miyako/ruri-v3-reranker-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/ruri-v3-reranker-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/ruri-v3-reranker-310m-onnx-int8)|`8192`|`768`|`25`
|[`jinaai/jina-reranker-v3`](https://huggingface.co/jinaai/jina-reranker-v3)|||[`598`](https://huggingface.co/keisuke-miyako/jina-reranker-v3-onnx-int8)|`131072`|`1024`|`28`|
|[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)|||[`598`](https://huggingface.co/keisuke-miyako/Qwen3-Reranker-0.6B-onnx-int8)|`32768`|`1024`|`28`|
|[`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B)|||[`4030`](https://huggingface.co/keisuke-miyako/Qwen3-Reranker-4B-onnx-int8)|`40960`|`2560`|`36`|
|[`zeroentropy/zerank-2`](https://huggingface.co/zeroentropy/zerank-2)|||[`4030`](https://huggingface.co/keisuke-miyako/zerank-2-onnx-int8)|`40960`|`2560`|`36`
|[`elyza/Llama-3-ELYZA-JP-8B`](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B)|[`6800`](https://huggingface.co/keisuke-miyako/Llama-3-ELYZA-JP-8B-onnx-int4)|`8192`|`4096`|`32`|
|[`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3`](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3)|[`6800`](https://huggingface.co/keisuke-miyako/Llama-3.1-Swallow-8B-Instruct-v0.3-onnx-int4)|`8192`|`4096`|`32`|
|[`Rakuten/RakutenAI-7B-chat`](https://huggingface.co/Rakuten/RakutenAI-7B-chat)|[`5290`](https://huggingface.co/keisuke-miyako/RakutenAI-7B-chat-onnx-int4)|`32768`|`4096`|`32`|
|[`Rakuten/RakutenAI-7B-instruct`](https://huggingface.co/Rakuten/RakutenAI-7B-instruct)|[`5290`](https://huggingface.co/keisuke-miyako/RakutenAI-7B-instruct-onnx-int4)|`32768`|`4096`|`32`|
|[`rinna/llama-3-youko-8b-instruct`](https://huggingface.co/rinna/llama-3-youko-8b-instruct)|[`6800`](https://huggingface.co/keisuke-miyako/llama-3-youko-8b-instruct-onnx-int4)|`8192`|`4096`|`32`|
|[`rinna/gemma-2-baku-2b-it`](https://huggingface.co/rinna/gemma-2-baku-2b-it)|[`4010`](https://huggingface.co/keisuke-miyako/gemma-2-baku-2b-it-onnx-int4)|`8192`|`2304`|`26`|
|[`rinna/youri-7b-instruction`](https://huggingface.co/rinna/youri-7b-instruction)|[`4660`](https://huggingface.co/keisuke-miyako/youri-7b-instruction-onnx-int4)|`4096`|`4096`|`32`|
|[`rinna/youri-7b-chat`](https://huggingface.co/rinna/youri-7b-chat)|[`4660`](https://huggingface.co/keisuke-miyako/youri-7b-chat-onnx-int4)|`4096`|`4096`|`32`|
|[`cyberagent/calm2-7b-chat`](https://huggingface.co/cyberagent/calm2-7b-chat)|[`5300`](https://huggingface.co/keisuke-miyako/calm2-7b-chat-onnx-int4)|`32768`|`4096`|`32`|

### Embedding

||`fp16`|`fp32`|`int8`|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`|`pooling`
|-|-:|-:|-:|-:|-:|-:|-:
|[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)|[`66`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-int8)|`512`|`384`|`12`|`cls`
|[`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)|[`278`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp16)|[`435`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp32)|[`116`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-int8)|`512`|`768`|`12`|`cls`
|[`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)|[`668`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp32)|[`335`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-int8)|`512`|`1024`|`24`|`cls`
|[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)|[`1130`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp32)|[`568`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-int8)|`8192`|`1024`|`24`|`cls`
|[`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2)|[`66`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-int8)|`512`|`384`|`12`|`mean`
|[`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2)|[`218`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp16)|[`435`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp32)|[`109`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-int8)|`512`|`768`|`12`|`mean`
|[`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2)|[`668`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp32)|[`335`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-int8)|`512`|`1024`|`24`|`mean`
|[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)|[`235`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp32)|[`118`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-int8)|`512`|`384`|`12`|`mean`
|[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)|[`555`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-int8)|`512`|`768`|`12`|`mean`
|[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)|[`1120`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp32)|[`560`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-int8)|`512`|`1024`|`24`|`mean`
|[`nomic-ai/nomic-embed-text-v1`](https://huggingface.co/nomic-ai/nomic-embed-text-v1)|[`274`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-fp16)|[`547`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-fp32)|[`138`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-int8)|`8192`|`768`|`12`|`mean`
|[`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)|[`274`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-fp16)|[`547`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-fp32)|[`138`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-int8)|`8192`|`768`|`12`|`mean`
|[`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)|[`66`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-int8)|`512`|`384`|`12`|`cls`
|[`Snowflake/snowflake-arctic-embed-l`](https://huggingface.co/Snowflake/snowflake-arctic-embed-l)|[`668`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-fp32)|[`336`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-int8)|`512`|`1024`|`24`|`cls`
|[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)|[`45`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-fp16)|[`90`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-fp32)|[`22`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-int8)|`512`|`384`|`6`|`mean`
|[`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)|[`66`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-int8)|`512`|`384`|`12`|`mean`
|[`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)|[`607`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-fp16)|[`1210`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-fp32)|[`309`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-int8)|`2048`|`768`|`24`|`mean`
|[`retrieva-jp/amber-base`](https://huggingface.co/retrieva-jp/amber-base)|[`264`](https://huggingface.co/keisuke-miyako/amber-base-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/amber-base-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/amber-base-onnx-int8)|`8192`|`512`|`19`|`meqn`
|[`retrieva-jp/amber-large`](https://huggingface.co/retrieva-jp/amber-large)|[`629`](https://huggingface.co/keisuke-miyako/amber-large-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/amber-large-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/amber-large-onnx-int8)|`8192`|`768`|`25`|`mean`
|[`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)|[`278`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-fp16)|[`556`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-fp32)|[`146`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-int8)|`8192`|`768`|`12`|`cls`
|[`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)|[`873`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-fp16)|[`1750`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-fp32)|[`445`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-int8)|`8192`|`1024`|`24`|`cls`
|[`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)|[`628`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-fp32)|[`340`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-int8)|`8192`|`768`|`12`|`cls`
|[`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)|[`298`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-fp16)|[`596`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-int8)|`8192`|`768`|`22`|`cls`
|[`sbintuitions/modernbert-ja-30m`](https://huggingface.co/sbintuitions/modernbert-ja-30m)|[`73`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-int8)|`8192`|`256`|`10`|`mean`
|[`sbintuitions/modernbert-ja-70m`](https://huggingface.co/sbintuitions/modernbert-ja-70m)|[`140`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-int8)|`8192`|`384`|`13`|`mean`
|[`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m)|[`264`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-int8)|`8192`|`512`|`19`|`mean`
|[`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m)|[`629`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-int8)|`8192`|`768`|`25`|`mean`
|[`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m)|[`73`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-int8)|`8192`|`256`|`10`|`mean`
|[`cl-nagoya/ruri-v3-70m`](https://huggingface.co/cl-nagoya/ruri-v3-70m)|[`140`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-int8)|`8192`|`384`|`13`|`mean`
|[`cl-nagoya/ruri-v3-130m`](https://huggingface.co/cl-nagoya/ruri-v3-130m)|[`264`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-int8)|`8192`|`512`|`19`|`mean`
|[`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m)|[`629`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-int8)|`8192`|`768`|`25`|`mean`
|[`cl-nagoya/ruri-base-v2`](https://huggingface.co/cl-nagoya/ruri-base-v2)|[`221`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-fp16)|[`442`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-fp32)|[`111`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-int8)|`512`|`768`|`12`|`mean`
|[`cl-nagoya/ruri-large-v2`](https://huggingface.co/cl-nagoya/ruri-large-v2)|[`673`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-fp16)|[`1350`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-fp32)|[`338`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-int8)|`512`|`1024`|`24`|`mean`
|[`ibm-granite/granite-embedding-small-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)|[`95`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-fp16)|[`190`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-fp32)|[`48`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-int8)|`8192`|`384`|`12`|`cls`
|[`ibm-granite/granite-embedding-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-english-r2)|[`298`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-fp16)|[`596`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-int8)|`8192`|`768`|`22`|`cls`
|[`ibm-granite/granite-embedding-30m-english`](https://huggingface.co/ibm-granite/granite-embedding-30m-english)|[`60`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-fp16)|[`120`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-fp32)|[`30`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-int8)|`512`|`384`|`6`|`cls`
|[`ibm-granite/granite-embedding-125m-english`](https://huggingface.co/ibm-granite/granite-embedding-125m-english)|[`248`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-fp16)|[`496`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-fp32)|[`125`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-int8)|`512`|`768`|`12`|`cls`
|[`ibm-granite/granite-embedding-107m-multilingual`](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual)|[`213`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-fp16)|[`427`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-fp32)|[`107`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-int8)|`512`|`384`|`6`|`cls`
|[`ibm-granite/granite-embedding-278m-multilingual`](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)|[`555`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-int8)|`512`|`768`|`12`|`cls`
|[`Alibaba-NLP/gte-Qwen2-1.5B-instruct`](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)|||[`1680`](https://huggingface.co/keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx-int8)|`32768`|`1536`|`28`|`last-token`
|[`Alibaba-NLP/gte-Qwen2-7B-instruct`](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)|||[`7210`](https://huggingface.co/keisuke-miyako/gte-Qwen2-7B-instruct-onnx-int8)|`32768`|`3584`|`28`|`last-token`
|[`sbintuitions/sarashina-embedding-v1-1b`](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b)|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v1-1b-onnx-int8)|`8192`|`1792`|`24`|`last-token`
|[`sbintuitions/sarashina-embedding-v2-1b`](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b)|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v2-1b-onnx-int8)|`8192`|`1792`|`24`|`last-token`
