# onnx-genai

ONNX Runtime GenAI Inference Engine

```
Usage:  onnx-genai -s -m chat_completion_model -e embedding_model -p port 

 -m path     : chat completion model
 -e path     : embedding model (pooling=mean)
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

## Converted ONNX Models

https://huggingface.co/collections/keisuke-miyako/onnx-runtime

- https://huggingface.co/keisuke-miyako/bge-small-en-v1.5-onnx
- https://huggingface.co/keisuke-miyako/bge-base-en-v1.5-onnx
- https://huggingface.co/keisuke-miyako/bge-large-en-v1.5-onnx
- https://huggingface.co/keisuke-miyako/bge-m3-onnx
- https://huggingface.co/keisuke-miyako/e5-small-v2-onnx
- https://huggingface.co/keisuke-miyako/e5-base-v2-onnx
- https://huggingface.co/keisuke-miyako/e5-large-v2-onnx
- https://huggingface.co/keisuke-miyako/multilingual-e5-small-onnx
- https://huggingface.co/keisuke-miyako/multilingual-e5-base-onnx
- https://huggingface.co/keisuke-miyako/multilingual-e5-large-onnx
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
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-large-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-multilingual-onnx
- https://huggingface.co/keisuke-miyako/universal-sentence-encoder-multilingual-large-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-small-english-r2-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-english-r2-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-30m-english-onnx
- https://huggingface.co/keisuke-miyako/granite-embedding-125m-english-onnx
- https://huggingface.co/keisuke-miyako/Granite-Embedding-107m-multilingual-onnx
- https://huggingface.co/keisuke-miyako/Granite-Embedding-278m-multilingual-onnx
