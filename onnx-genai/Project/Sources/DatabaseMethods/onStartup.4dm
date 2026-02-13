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
	
	$folder:=$homeFolder.folder("Qwen3-4B-Instruct-2507")
	$path:="Qwen3-4B-Instruct-2507-onnx-int4-cpu"
	$URL:="keisuke-miyako/Qwen3-4B-Instruct-2507-onnx-int4-cpu"
	
	//$folder:=$homeFolder.folder("Qwen3-4B-Thinking-2507")
	//$path:="Qwen3-4B-Thinking-2507-onnx-int4-cpu"
	//$URL:="keisuke-miyako/Qwen3-4B-Thinking-2507-onnx-int4-cpu"
	
	//$folder:=$homeFolder.folder("translategemma-4b-it")
	//$path:="translategemma-4b-it-onnx-int4-cpu"
	//$URL:="keisuke-miyako/translategemma-4b-it-onnx-int4-cpu"
	
	//$folder:=$homeFolder.folder("gemma-3-4b-it")
	//$path:="gemma-3-4b-it-onnx-int4-cpu"
	//$URL:="keisuke-miyako/gemma-3-4b-it-onnx-int4-cpu"
	
	//$folder:=$homeFolder.folder("Ministral-3-3B-Instruct")
	//$path:="Ministral-3-3B-Instruct-2512-ONNX"
	//$URL:="mistralai/Ministral-3-3B-Instruct-2512-ONNX"
	
	//$chat:=cs.event.huggingface.new($folder; $URL; $path; "chat.completion")
	
	$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
	
	Case of 
		: (False:C215)  //e2e
			$folder:=$homeFolder.folder("universal-sentence-encoder-multilingual")
			$path:="universal-sentence-encoder-multilingual-onnx"
			$URL:="keisuke-miyako/universal-sentence-encoder-multilingual-onnx"
			$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
			$options:={pooling: "e2e"}
	End case 
	
	If (False:C215)  //embedding
		
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
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model.onnx")
		
		$folder:=$homeFolder.folder("gte-Qwen2-1.5B-instruct")
		$path:="gte-Qwen2-1.5B-instruct-onnx-int8"
		$URL:="keisuke-miyako/gte-Qwen2-1.5B-instruct-onnx-int8"
		$options:={pooling: "last-token"}
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
		
		$huggingfaces:=cs:C1710.event.huggingfaces.new([$huggingface])
		
		$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
		
	End if 
	
	If (True:C214)  //rerank
		
		$homeFolder:=Folder:C1567(fk home folder:K87:24).folder(".ONNX")
		$port:=8081
		
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
		
		$huggingface:=cs:C1710.event.huggingface.new($folder; $URL; $path; "rerank"; "model.onnx")
		$huggingfaces:=cs:C1710.event.huggingfaces.new([$huggingface])
		
		$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
		
	End if 
	
End if 