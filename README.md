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

## OpenAI Compatible Endpoints

- `/v1/models`
- `/v1/chat/completions`
- `/v1/embeddings`

## Cohere Compatible Endpoints

- `/v1/rerank`

## Converted ONNX Models

### Rerank

|Model|fp16|fp32|int8
|-|-:|-:|-:
|`cross-encoder/ms-marco-MiniLM-L6-v2`||[`91`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp32)|
|`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`|[`235`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp32)
|`BAAI/bge-reranker-v2-m3`|[`1140`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp32)
|`BAAI/bge-reranker-base`|[`556`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp32)
|`BAAI/bge-reranker-large`|[`1120`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp32)
|`jinaai/jina-reranker-v1-turbo-en`||[`151`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp32)
|`mixedbread-ai/mxbai-rerank-xsmall-v1`||[`284`](https://huggingface.co/keisuke-miyako/mxbai-rerank-xsmall-v1-onnx-fp32)
|`ibm-granite/granite-embedding-reranker-english-r2`|[`299`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp16)|[`599`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp32)

### Embedding

|Model|fp16|fp32|int8
|-|-:|-:|-:
|`BAAI/bge-small-en-v1.5`||[`133`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-int8)
|`BAAI/bge-base-en-v1.5`||[`435`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp32)|[`116`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-int8)
|`BAAI/bge-large-en-v1.5`||[`1340`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp32)|[`335`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-int8)
|`BAAI/bge-m3`|[`1130`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp32)
|`intfloat/e5-small-v2`||[`133`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp32)
|`intfloat/e5-base-v2`||[`435`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp32)
|`intfloat/e5-large-v2`|[`668`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp32)
|`intfloat/multilingual-e5-small`||[`470`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp32)
|`intfloat/multilingual-e5-base`|[`555`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp32)
|`intfloat/multilingual-e5-large`|[`1120`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp32)
|`nomic-ai/nomic-embed-text-v1`
|`nomic-ai/nomic-embed-text-v1.5`
|`Snowflake/snowflake-arctic-embed-s`
|`Snowflake/snowflake-arctic-embed-l`|
|`sentence-transformers/all-MiniLM-L6-v2`
|`sentence-transformers/all-MiniLM-L12-v2`
|`google/embeddinggemma-300m`
|`retrieva-jp/amber-base`
|`retrieva-jp/amber-large`
|`Alibaba-NLP/gte-base-en-v1.5`
|`Alibaba-NLP/gte-large-en-v1.5`
|`Alibaba-NLP/gte-multilingual-base`
|`Alibaba-NLP/gte-modernbert-base`
|`Alibaba-NLP/gte-Qwen2-1.5B-instruct`|||[`1680`](https://huggingface.co/keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx-int8)
|`Alibaba-NLP/gte-Qwen2-7B-instruct`|||[`7210`](https://huggingface.co/keisuke-miyako/gte-Qwen2-7B-instruct-onnx-int8)|
|`sbintuitions/modernbert-ja-30m`
|`sbintuitions/modernbert-ja-70m`
|`sbintuitions/modernbert-ja-130m`
|`sbintuitions/modernbert-ja-310m`
|`cl-nagoya/ruri-v3-30m`
|`cl-nagoya/ruri-v3-70m`
|`cl-nagoya/ruri-v3-130m`
|`cl-nagoya/ruri-v3-310m`
|`cl-nagoya/ruri-base-v2`
|`cl-nagoya/ruri-large-v2`
|`ibm-granite/granite-embedding-small-english-r2`
|`ibm-granite/granite-embedding-english-r2`
|`ibm-granite/granite-embedding-30m-english`
|`ibm-granite/granite-embedding-125m-english`
|`ibm-granite/granite-embedding-107m-multilingual`
|`ibm-granite/granite-embedding-278m-multilingual`
|`sbintuitions/sarashina-embedding-v1-1b`|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v1-1b-onnx-int8)
|`sbintuitions/sarashina-embedding-v2-1b`|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v2-1b-onnx-int8)
