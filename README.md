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
|`BAAI/bge-reranker-base`|`556`|`1110`
|`BAAI/bge-reranker-large`||`2240`
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
|`gte-Qwen2-1.5B-instruct`|||`1680`
|`gte-Qwen2-7B-instruct`|||`7210`|
|`sarashina-embedding-v1-1b`|||`1230`
|`sarashina-embedding-v2-1b`|||`1230`
|`universal-sentence-encoder`|||`589`
|`universal-sentence-encoder-large`|||`1030`
|`universal-sentence-encoder-multilingual`|||`279`
|`universal-sentence-encoder-multilingual-large`|||`340`
