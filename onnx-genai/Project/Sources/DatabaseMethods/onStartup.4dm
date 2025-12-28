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
	$event.onData:=Formula(MESSAGE(This.file.fullName+":"+String((This.range.end/This.range.length)*100; "###.00%")))
	$event.onResponse:=Formula:C1597(LOG EVENT:C667(Into 4D debug message:K38:5; This:C1470.file.fullName+":download complete"))
	$event.onResponse:=Formula(MESSAGE(This.file.fullName+":download complete"))
	$event.onTerminate:=Formula:C1597(LOG EVENT:C667(Into 4D debug message:K38:5; (["process"; $1.pid; "terminated!"].join(" "))))
	
	$port:=8080
	
	$folder:=$homeFolder.folder("Phi-3.5-mini-instruct")
	$URL:="microsoft/Phi-3.5-mini-instruct"
	$chat:=cs:C1710.event.huggingface.new($folder; $URL; "chat.completion")
	
	$folder:=$homeFolder.folder("all-MiniLM-L6-v2")
	$URL:="ONNX-models/all-MiniLM-L6-v2-ONNX"
	$URL:="https://huggingface.co/ONNX-models/all-MiniLM-L6-v2-ONNX/main"
	$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; "embedding")
	
	$options:={}
	var $huggingfaces : cs:C1710.event.huggingfaces
	$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
	
	$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $options; $event)
	
End if 