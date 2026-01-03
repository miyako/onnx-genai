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
	
	Case of 
		: (True:C214)
			$folder:=$homeFolder.folder("Baguettotron")
			$path:="keisuke-miyako/Baguettotron-onnx"
			$URL:="keisuke-miyako/Baguettotron-onnx"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			
			$options:={c: "qwen"}
			
		: (False:C215)
			$folder:=$homeFolder.folder("microsoft/Phi-3.5-mini-instruct")
			$path:="cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
			$URL:="https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			
			$folder:=$homeFolder.folder("all-MiniLM-L6-v2")
			$path:=""
			$URL:="ONNX-models/all-MiniLM-L6-v2-ONNX"
			$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding")
			
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
			
			$options:={c: "phi"}
	End case 
	
	$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
	
End if 