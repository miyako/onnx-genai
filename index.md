---
layout: default
---

![version](https://img.shields.io/badge/version-20%2B-E23089)
![platform](https://img.shields.io/static/v1?label=platform&message=mac-intel%20|%20mac-arm%20|%20win-64&color=blue)
[![license](https://img.shields.io/github/license/miyako/llama-cpp)](LICENSE)
![downloads](https://img.shields.io/github/downloads/miyako/llama-cpp/total)

# Use llama.cpp from 4D

#### Abstract

[**llama.cpp**](https://github.com/ggml-org/llama.cpp) is an open-source project that allows you to run Meta's LLaMA language models locally on CPUs without heavy frameworks like PyTorch or TensorFlow. Essentially, it’s a **lightweight C++ implementation optimized for inference**.

#### Usage

Instantiate `cs.llama.llama` in your *On Startup* database method:

```4d
var $llama : cs.llama.llama

If (False)
    $llama:=cs.llama.llama.new()  //default
Else 
    var $homeFolder : 4D.Folder
    $homeFolder:=Folder(fk home folder).folder(".llama-cpp")
    var $file : 4D.File
    var $URL : Text
    var $port : Integer
    
    var $event : cs.event.event
    $event:=cs.event.event.new()
    /*
        Function onError($params : Object; $error : cs.event.error)
        Function onSuccess($params : Object; $models : cs.event.models)
        Function onData($request : 4D.HTTPRequest; $event : Object)
        Function onResponse($request : 4D.HTTPRequest; $event : Object)
        Function onTerminate($worker : 4D.SystemWorker; $params : Object)
    */
    
    $event.onError:=Formula(ALERT($2.message))
    $event.onSuccess:=Formula(ALERT($2.models.extract("name").join(",")+" loaded!"))
    $event.onData:=Formula(LOG EVENT(Into 4D debug message; "download:"+String((This.range.end/This.range.length)*100; "###.00%")))
    $event.onResponse:=Formula(LOG EVENT(Into 4D debug message; "download complete"))
    $event.onTerminate:=Formula(LOG EVENT(Into 4D debug message; (["process"; $1.pid; "terminated!"].join(" "))))
    
    /*
        embeddings
    */
    
    $file:=$homeFolder.file("nomic-embed-text-v1.Q8_0.gguf")
    $URL:="https://huggingface.co/nomic-ai/nomic-embed-text-v1-GGUF/resolve/main/nomic-embed-text-v1.Q8_0.gguf"
    $port:=8082
    $llama:=cs.llama.llama.new($port; $file; $URL; {\
    ctx_size: 2048; \
    batch_size: 2048; \
    threads: 4; \
    threads_batch: 4; \
    threads_http: 4; \
    temp: 0.7; \
    top_k: 40; \
    top_p: 0.9; \
    log_disable: True; \
    repeat_penalty: 1.1; \
    n_gpu_layers: -1}; $event)
    
    /*
        chat completion (with images)
    */
    
    $file:=$homeFolder.file("Qwen2-VL-2B-Instruct-Q4_K_M")
    $URL:="https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
    $port:=8083
    $llama:=cs.llama.llama.new($port; $file; $URL; {\
    ctx_size: 2048; \
    batch_size: 2048; \
    threads: 4; \
    threads_batch: 4; \
    threads_http: 4; \
    temp: 0.7; \
    top_k: 40; \
    top_p: 0.9; \
    log_disable: True; \
    repeat_penalty: 1.1; \
    n_gpu_layers: -1}; $event)
    
End if   
```

Unless the server is already running (in which case the costructor does nothing), the following procedure runs in the background:

1. The specified model is downloaded via HTTP
2. The `llama-server` program is started

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

```

```

```

```
hf download LightEmbed/baai-bge-base-en-v1.5-onnx --local-dir .
```




Use [optimum-cli](https://github.com/huggingface/optimum) to convert a specific model to ONNX:

```
optimum-cli export onnx --model BAAI/bge-base-en-v1.5 onnx_output_dir/
```



Finally to terminate the server:

```4d

```

#### AI Kit compatibility

The API is compatibile with [Open AI](https://platform.openai.com/docs/api-reference/embeddings). 

|Class|API|Availability|
|-|-|:-:|
|Models|`/v1/models`|✅|
|Chat|`/v1/chat/completions`|✅|
|Images|`/v1/images/generations`||
|Moderations|`/v1/moderations`||
|Embeddings|`/v1/embeddings`|✅|
|Files|`/v1/files`||
