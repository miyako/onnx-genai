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
|`cross-encoder/ms-marco-MiniLM-L6-v2`|[`45`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp16)|[`91`](https://huggingface.co/keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp32)|
|`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`|[`235`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp32)
|`BAAI/bge-reranker-v2-m3`|[`1140`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-reranker-v2-m3-onnx-fp32)
|`BAAI/bge-reranker-base`|[`556`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/bge-reranker-base-onnx-fp32)
|`BAAI/bge-reranker-large`|[`1120`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/bge-reranker-large-onnx-fp32)
|`jinaai/jina-reranker-v1-turbo-en`|[`75`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp16)|[`151`](https://huggingface.co/keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp32)
|`mixedbread-ai/mxbai-rerank-xsmall-v1`|⚠️[`142`](https://huggingface.co/keisuke-miyako/mxbai-rerank-xsmall-v1-onnx-fp16)|[`284`](https://huggingface.co/keisuke-miyako/mxbai-rerank-xsmall-v1-onnx-fp32)
|`ibm-granite/granite-embedding-reranker-english-r2`|[`299`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp16)|[`599`](https://huggingface.co/keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp32)

### Embedding

|Model|fp16|fp32|int8
|-|-:|-:|-:
|`BAAI/bge-small-en-v1.5`|[`66`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx-int8)
|`BAAI/bge-base-en-v1.5`|[`278`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp16)|[`435`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-fp32)|[`116`](https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx-int8)
|`BAAI/bge-large-en-v1.5`|[`668`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-fp32)|[`335`](https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx-int8)
|`BAAI/bge-m3`|[`1130`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp16)|[`2270`](https://huggingface.co/keisuke-miyako/bge-m3-onnx-fp32)
|`intfloat/e5-small-v2`|[`66`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/e5-small-v2-onnx-fp32)
|`intfloat/e5-base-v2`|[`218`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp16)|[`435`](https://huggingface.co/keisuke-miyako/e5-base-v2-onnx-fp32)
|`intfloat/e5-large-v2`|[`668`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/e5-large-v2-onnx-fp32)
|`intfloat/multilingual-e5-small`|[`235`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp16)|[`470`](https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx-fp32)
|`intfloat/multilingual-e5-base`|[`555`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx-fp32)
|`intfloat/multilingual-e5-large`|[`1120`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp16)|[`2240`](https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx-fp32)
|`nomic-ai/nomic-embed-text-v1`|[`274`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-fp16)|[`547`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-fp32)|[`138`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1-onnx-int8)
|`nomic-ai/nomic-embed-text-v1.5`|[`274`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-fp16)|[`547`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-fp32)|[`138`](https://huggingface.co/keisuke-miyako/nomic-embed-text-v1.5-onnx-int8)
|`Snowflake/snowflake-arctic-embed-s`|[`66`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-s-onnx-int8)
|`Snowflake/snowflake-arctic-embed-l`|[`668`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-fp16)|[`1340`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-fp32)|[`336`](https://huggingface.co/keisuke-miyako/snowflake-arctic-embed-l-onnx-int8)
|`sentence-transformers/all-MiniLM-L6-v2`|[`45`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-fp16)|[`90`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-fp32)|[`22`](https://huggingface.co/keisuke-miyako/all-MiniLM-L6-v2-onnx-int8)
|`sentence-transformers/all-MiniLM-L12-v2`|[`66`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16)|[`133`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-fp32)|[`33`](https://huggingface.co/keisuke-miyako/all-MiniLM-L12-v2-onnx-int8)
|`google/embeddinggemma-300m`|[`607`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-fp16)|[`1210`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-fp32)|[`309`](https://huggingface.co/keisuke-miyako/embeddinggemma-300m-onnx-int8)
|`retrieva-jp/amber-base`|[`264`](https://huggingface.co/keisuke-miyako/amber-base-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/amber-base-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/amber-base-onnx-int8)
|`retrieva-jp/amber-large`|[`629`](https://huggingface.co/keisuke-miyako/amber-large-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/amber-large-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/amber-large-onnx-int8)
|`Alibaba-NLP/gte-base-en-v1.5`|[`278`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-fp16)|[`556`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-fp32)|[`146`](https://huggingface.co/keisuke-miyako/gte-base-en-v1.5-onnx-int8)
|`Alibaba-NLP/gte-large-en-v1.5`|[`873`](keisuke-miyako/gte-large-en-v1.5-onnx-fp16)|[`1750`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-fp32)|[`445`](https://huggingface.co/keisuke-miyako/gte-large-en-v1.5-onnx-int8)
|`Alibaba-NLP/gte-multilingual-base`|[`628`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-fp32)|[`340`](https://huggingface.co/keisuke-miyako/gte-multilingual-base-onnx-int8)
|`Alibaba-NLP/gte-modernbert-base`|[`298`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-fp16)|[`596`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/gte-modernbert-base-onnx-int8)
|`Alibaba-NLP/gte-Qwen2-1.5B-instruct`|||[`1680`](https://huggingface.co/keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx-int8)
|`Alibaba-NLP/gte-Qwen2-7B-instruct`|||[`7210`](https://huggingface.co/keisuke-miyako/gte-Qwen2-7B-instruct-onnx-int8)|
|`sbintuitions/modernbert-ja-30m`|[`73`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/modernbert-ja-30m-onnx-int8)
|`sbintuitions/modernbert-ja-70m`|[`140`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/modernbert-ja-70m-onnx-int8)
|`sbintuitions/modernbert-ja-130m`|[`264`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/modernbert-ja-130m-onnx-int8)
|`sbintuitions/modernbert-ja-310m`|[`629`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/modernbert-ja-310m-onnx-int8)
|`cl-nagoya/ruri-v3-30m`|[`73`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-fp16)|[`147`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-fp32)|[`37`](https://huggingface.co/keisuke-miyako/ruri-v3-30m-onnx-int8)
|`cl-nagoya/ruri-v3-70m`|[`140`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-fp16)|[`280`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-fp32)|[`70`](https://huggingface.co/keisuke-miyako/ruri-v3-70m-onnx-int8)
|`cl-nagoya/ruri-v3-130m`|[`264`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-fp16)|[`529`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-fp32)|[`133`](https://huggingface.co/keisuke-miyako/ruri-v3-130m-onnx-int8)
|`cl-nagoya/ruri-v3-310m`|[`629`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-fp16)|[`1260`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-fp32)|[`316`](https://huggingface.co/keisuke-miyako/ruri-v3-310m-onnx-int8)
|`cl-nagoya/ruri-base-v2`|[`221`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-fp16)|[`442`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-fp32)|[`111`](https://huggingface.co/keisuke-miyako/ruri-base-v2-onnx-int8)
|`cl-nagoya/ruri-large-v2`|[`673`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-fp16)|[`1350`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-fp32)|[`338`](https://huggingface.co/keisuke-miyako/ruri-large-v2-onnx-int8)
|`ibm-granite/granite-embedding-small-english-r2`|[`95`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-fp16)|[`190`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-fp32)|[`48`](https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx-int8)
|`ibm-granite/granite-embedding-english-r2`|[`298`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-fp16)|[`596`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-fp32)|[`150`](https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx-int8)
|`ibm-granite/granite-embedding-30m-english`|[`60`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-fp16)|[`120`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-fp32)|[`30`](https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx-int8)
|`ibm-granite/granite-embedding-125m-english`|[`248`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-fp16)|[`496`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-fp32)|[`125`](https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx-int8)
|`ibm-granite/granite-embedding-107m-multilingual`|[`213`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-fp16)|[`427`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-fp32)|[`107`](https://huggingface.co/keisuke-miyako/granite-embedding-107m-multilingual-onnx-int8)
|`ibm-granite/granite-embedding-278m-multilingual`|[`555`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-fp16)|[`1110`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-fp32)|[`278`](https://huggingface.co/keisuke-miyako/granite-embedding-278m-multilingual-onnx-int8)
|`sbintuitions/sarashina-embedding-v1-1b`|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v1-1b-onnx-int8)
|`sbintuitions/sarashina-embedding-v2-1b`|||[`1230`](https://huggingface.co/keisuke-miyako/sarashina-embedding-v2-1b-onnx-int8)
