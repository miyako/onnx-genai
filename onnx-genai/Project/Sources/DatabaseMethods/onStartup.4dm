var $ONNX : cs:C1710.ONNX

If (False:C215)
	$ONNX:=cs:C1710.ONNX.new()  //default
Else 
	var $homeFolder : 4D:C1709.Folder
	$homeFolder:=Folder:C1567(fk home folder:K87:24).folder(".ONNX")
	var $file : 4D:C1709.File
	var $URL : Text
	var $port : Integer
	
	var $event : cs:C1710.event.event
	$event:=cs:C1710.event.event.new()
/*
Function onError($params : Object; $error : cs.event.error)
Function onSuccess($params : Object; $models : cs.event.models)
Function onData($request : 4D.HTTPRequest; $event : Object)
Function onResponse($request : 4D.HTTPRequest; $event : Object)
Function onTerminate($worker : 4D.SystemWorker; $params : Object)
*/
	
	$event.onError:=Formula:C1597(ALERT:C41($2.message))
	$event.onSuccess:=Formula:C1597(ALERT:C41($2.models.extract("name").join(",")+" loaded!"))
	$event.onData:=Formula:C1597(LOG EVENT:C667(Into 4D debug message:K38:5; This:C1470.file.fullName+":"+String:C10((This:C1470.range.end/This:C1470.range.length)*100; "###.00%")))
	$event.onData:=Formula:C1597(MESSAGE:C88(This:C1470.file.fullName+":"+String:C10((This:C1470.range.end/This:C1470.range.length)*100; "###.00%")))
	$event.onResponse:=Formula:C1597(LOG EVENT:C667(Into 4D debug message:K38:5; This:C1470.file.fullName+":download complete"))
	$event.onResponse:=Formula:C1597(MESSAGE:C88(This:C1470.file.fullName+":download complete"))
	$event.onTerminate:=Formula:C1597(LOG EVENT:C667(Into 4D debug message:K38:5; (["process"; $1.pid; "terminated!"].join(" "))))
	
	$port:=8080
	
	$options:={}
	var $huggingfaces : cs:C1710.event.huggingfaces
	
	If (False:C215)
		$folder:=$homeFolder.folder("Phi-4-mini-reasoning")
		$path:="Phi-4-mini-reasoning-onnx-int4"
		$URL:="keisuke-miyako/Phi-4-mini-reasoning-onnx-int4"
		
		$folder:=$homeFolder.folder("Qwen3-0.6B")
		$path:="Qwen3-0.6B-onnx-int4"
		$URL:="keisuke-miyako/Qwen3-0.6B-onnx-int4"
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion"; "model.onnx")
		$huggingfaces:=cs:C1710.event.huggingfaces.new([$huggingface])
		
		$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
	End if 
	
	If (True:C214)  //embedding 
		
		$folder:=$homeFolder.folder("universal-sentence-encoder-multilingual")
		$path:="universal-sentence-encoder-multilingual-onnx"
		$URL:="keisuke-miyako/universal-sentence-encoder-multilingual-onnx"
		$options:={pooling: "e2e"}
		
		$folder:=$homeFolder.folder("bge-small-en-v1.5")
		$path:="bge-small-en-v1.5-onnx-fp16"
		$URL:="keisuke-miyako/bge-small-en-v1.5-onnx-fp16"
		$options:={pooling: "cls"}
		
		$folder:=$homeFolder.folder("bge-base-en-v1.5")
		$path:="bge-base-en-v1.5-onnx-fp16"
		$URL:="keisuke-miyako/bge-base-en-v1.5-onnx-fp16"
		$options:={pooling: "cls"}
		
		$folder:=$homeFolder.folder("bge-large-en-v1.5")
		$path:="bge-large-en-v1.5-onnx-fp16"
		$URL:="keisuke-miyako/bge-large-en-v1.5-onnx-fp16"
		$options:={pooling: "cls"}
		
		$folder:=$homeFolder.folder("bge-m3")
		$path:="bge-m3-onnx-fp16"
		$URL:="keisuke-miyako/bge-m3-onnx-fp16"
		$options:={pooling: "cls"}
		
		$folder:=$homeFolder.folder("multilingual-e5-large")
		$path:="multilingual-e5-large-onnx-fp16"
		$URL:="keisuke-miyako/multilingual-e5-large-onnx-fp16"
		$options:={pooling: "mean"}
		
		$folder:=$homeFolder.folder("e5-small")
		$path:="e5-small-v2-onnx-fp16"
		$URL:="keisuke-miyako/e5-small-v2-onnx-fp16"
		$options:={pooling: "mean"}
		
		$folder:=$homeFolder.folder("e5-base")
		$path:="e5-base-v2-onnx-fp16"
		$URL:="keisuke-miyako/e5-base-v2-onnx-fp16"
		$options:={pooling: "mean"}
		
		$folder:=$homeFolder.folder("multilingual-e5-small")
		$path:="multilingual-e5-small-onnx-fp16"
		$URL:="keisuke-miyako/multilingual-e5-small-onnx-fp16"
		$options:={pooling: "mean"}
		
		$folder:=$homeFolder.folder("gte-Qwen2-1.5B-instruct")
		$path:="gte-Qwen2-1.5B-instruct-onnx-int8"
		$URL:="keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx-int8"
		$options:={pooling: "last-token"}
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
		
		$folder:=$homeFolder.folder("Kokoro-82M")
		$path:="Kokoro-82M-onnx-f32"
		$URL:="keisuke-miyako/Kokoro-82M-onnx-f32"
		$options:={}
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "tts"; "model.onnx")
		$huggingfaces:=cs:C1710.event.huggingfaces.new([$huggingface])
		
		$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
		
	End if 
	
	If (False:C215)  //rerank
		
		$homeFolder:=Folder:C1567(fk home folder:K87:24).folder(".ONNX")
		$port:=8080
		
		$folder:=$homeFolder.folder("ms-marco-MiniLM-L6-v2")  //where to keep the repo
		$path:="ms-marco-MiniLM-L6-v2-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/ms-marco-MiniLM-L6-v2-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("mmarco-mMiniLMv2-L12-H384-v1")  //where to keep the repo
		$path:="mmarco-mMiniLMv2-L12-H384-v1-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/mmarco-mMiniLMv2-L12-H384-v1-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("jina-reranker-v1-turbo-en")  //where to keep the repo
		$path:="jina-reranker-v1-turbo-en-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("bge-reranker-base")  //where to keep the repo
		$path:="bge-reranker-base-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/bge-reranker-base-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("jina-reranker-v1-turbo-en")  //where to keep the repo
		$path:="jina-reranker-v1-turbo-en-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/jina-reranker-v1-turbo-en-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("granite-embedding-reranker-english-r2")  //where to keep the repo
		$path:="granite-embedding-reranker-english-r2-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/granite-embedding-reranker-english-r2-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("bge-reranker-large")  //where to keep the repo
		$path:="bge-reranker-large-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/bge-reranker-large-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("bge-reranker-v2-m3")  //where to keep the repo
		$path:="bge-reranker-v2-m3-onnx-fp16"  //path to the file
		$URL:="keisuke-miyako/bge-reranker-v2-m3-onnx-fp16"  //path to the repo
		
		$folder:=$homeFolder.folder("bge-reranker-v2-m3")  //where to keep the repo
		$path:="bge-reranker-v2-m3-onnx-int8"  //path to the file
		$URL:="keisuke-miyako/bge-reranker-v2-m3-onnx-int8"  //path to the repo
		
		//$folder:=$homeFolder.folder("Qwen3-Reranker-0.6B")  //where to keep the repo
		//$path:="Qwen3-Reranker-0.6B-onnx-int8"  //path to the file
		//$URL:="keisuke-miyako/Qwen3-Reranker-0.6B-onnx-int8"  //path to the repo
		
		//$folder:=$homeFolder.folder("jina-reranker-v3")  //where to keep the repo
		//$path:="jina-reranker-v3-onnx-int8"  //path to the file
		//$URL:="keisuke-miyako/jina-reranker-v3-onnx-int8"  //path to the repo
		
		//$folder:=$homeFolder.folder("zerank-2")  //where to keep the repo
		//$path:="zerank-2-onnx-int8"  //path to the file
		//$URL:="keisuke-miyako/zerank-2-onnx-int8"  //path to the repo
		
		//$folder:=$homeFolder.folder("Qwen3-Reranker-4B")  //where to keep the repo
		//$path:="Qwen3-Reranker-4B-onnx-int8"  //path to the file
		//$URL:="keisuke-miyako/Qwen3-Reranker-4B-onnx-int8"  //path to the repo
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "rerank"; "model_quantized.onnx")
		$huggingfaces:=cs:C1710.event.huggingfaces.new([$huggingface])
		
		$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
		
	End if 
	
End if 