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
|`cross-encoder/ms-marco-MiniLM-L6-v2`||`91`|
|`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`|`235`|`470`
|`BAAI/bge-reranker-v2-m3`||`2270`
|`BAAI/bge-reranker-large`||`2240`
|`BAAI/bge-reranker-base`|`556`|`1110`
|`jinaai/jina-reranker-v1-turbo-en`||`151`
|`mixedbread-ai/mxbai-rerank-xsmall-v1`||`284`
|`ibm-granite/granite-embedding-reranker-english-r2`|`299`|`599`

### Embedding

|Model|fp16|fp32|int8
|-|-:|-:|-:
|`BAAI/bge-small-en-v1.5`|
|`BAAI/bge-base-en-v1.5`
|`BAAI/bge-large-en-v1.5`
|`BAAI/bge-m3`||`2270`
|`intfloat/e5-small-v2`||`133`
|`intfloat/e5-base-v2`||`435`
|`intfloat/e5-large-v2`|`668`|`1340`
|`intfloat/multilingual-e5-small`||`470`
|`intfloat/multilingual-e5-base`|`555`|`1110`
|`intfloat/multilingual-e5-large`||`2240`

- https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx
- https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx
- https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx
- https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx
- https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx
- https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx
- https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx
- https://huggingface.co/keisuke-miyako/amber-base-onnx
- https://huggingface.co/keisuke-miyako/amber-large-onnx
- https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx
- https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx
- https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx
- https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx
- https://huggingface.co/keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx
- https://huggingface.co/keisuke-miyako/gte-Qwen2-7B-instruct-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-large-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-multilingual-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-multilingual-large-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx
- https://huggingface.co/keisuke-miyako/sarashina-embedding-v1-1b-onnx
- https://huggingface.co/keisuke-miyako/sarashina-embedding-v2-1b-onnx
- https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx
- https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx
- https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx
- https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx
- https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx
- https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx
- https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx
- https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx
- https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx
- https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx
