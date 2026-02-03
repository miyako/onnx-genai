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
	
	$folder:=$homeFolder.folder("Qwen3-4B-Thinking-2507")
	$path:="Qwen3-4B-Thinking-2507-onnx-int4-cpu"
	$URL:="keisuke-miyako/Qwen3-4B-Thinking-2507-onnx-int4-cpu"
	
	//$folder:=$homeFolder.folder("translategemma-4b-it")
	//$path:="translategemma-4b-it-onnx-int4-cpu"
	//$URL:="keisuke-miyako/translategemma-4b-it-onnx-int4-cpu"
	
	//$folder:=$homeFolder.folder("gemma-3-4b-it")
	//$path:="gemma-3-4b-it-onnx-int4-cpu"
	//$URL:="keisuke-miyako/gemma-3-4b-it-onnx-int4-cpu"
	
	$folder:=$homeFolder.folder("Ministral-3-3B-Instruct")
	$path:="Ministral-3-3B-Instruct-2512-ONNX"
	$URL:="mistralai/Ministral-3-3B-Instruct-2512-ONNX"
	
	$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
	
	$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
	
	Case of 
		: (True:C214)  //mean
			$folder:=$homeFolder.folder("multilingual-e5-base")
			$path:="multilingual-e5-base-onnx"
			$URL:="keisuke-miyako/multilingual-e5-base-onnx"
			$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
			$options:={pooling: "mean"}
		: (True:C214)  //e2e
			$folder:=$homeFolder.folder("universal-sentence-encoder-multilingual")
			$path:="universal-sentence-encoder-multilingual-onnx"
			$URL:="keisuke-miyako/universal-sentence-encoder-multilingual-onnx"
			$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
			$options:={pooling: "e2e"}
	End case 
	
	$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
	
End if 