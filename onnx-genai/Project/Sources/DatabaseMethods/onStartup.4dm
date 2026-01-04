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
		: (False:C215)  //not working
			$chat_template:=File:C1566("/RESOURCES/jinja/CroissantLLM.jinja").getText()
			$folder:=$homeFolder.folder("CroissantLLMChat-v0.1-onnx-int4-cpu")
			$path:="keisuke-miyako/CroissantLLMChat-v0.1-onnx-int4-cpu"
			$URL:="keisuke-miyako/CroissantLLMChat-v0.1-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:=File:C1566("/RESOURCES/jinja/SmolLM.jinja").getText()
			$folder:=$homeFolder.folder("SmolLM2-1.7B-onnx-int4-cpu")
			$path:="keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu"
			$URL:="keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //not working
			$chat_template:=File:C1566("/RESOURCES/jinja/gemma3.jinja").getText()
			$folder:=$homeFolder.folder("gemma-3-1b-it-onnx-int4-cpu")
			$path:="keisuke-miyako/gemma-3-1b-it-onnx-int4-cpu"
			$URL:="keisuke-miyako/gemma-3-1b-it-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //not working
			$chat_template:=File:C1566("/RESOURCES/jinja/Ministral-3-3B-Instruct-2512.jinja").getText()
			$folder:=$homeFolder.folder("Ministral-3.3B-onnx-int4-cpu")
			$path:="keisuke-miyako/Ministral-3.3B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Ministral-3.3B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:=File:C1566("/RESOURCES/jinja/gemma2.jinja").getText()
			$folder:=$homeFolder.folder("gemma-2-2b-it-onnx-int4-cpu")
			$path:="keisuke-miyako/gemma-2-2b-it-onnx-int4-cpu"
			$URL:="keisuke-miyako/gemma-2-2b-it-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:=File:C1566("/RESOURCES/jinja/Qwen3.jinja").getText()
			$folder:=$homeFolder.folder("Qwen3-1.7B-onnx-int4-cpu")
			$path:="keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:=File:C1566("/RESOURCES/jinja/Qwen2.5.jinja").getText()
			$folder:=$homeFolder.folder("Qwen2.5-1.5B-onnx-int4-cpu")
			$path:="keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:=File:C1566("/RESOURCES/jinja/Baguettotron.jinja").getText()
			$folder:=$homeFolder.folder("Baguettotron-onnx-int4-cpu")
			$path:="keisuke-miyako/Baguettotron-onnx-int4-cpu"
			$URL:="keisuke-miyako/Baguettotron-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:=File:C1566("/RESOURCES/jinja/EuroLLM.jinja").getText()
			$folder:=$homeFolder.folder("EuroLLM-1.7B-Instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
			//make sure to set max_tokens<2048 in request 
		: (True:C214)
			$chat_template:=File:C1566("/RESOURCES/jinja/Llama-3.2.jinja").getText()
			$folder:=$homeFolder.folder("Llama-3.2-1B-Instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			
			$folder:=$homeFolder.folder("bge-small-en-v1.5-onnx")
			$path:="keisuke-miyako/bge-small-en-v1.5-onnx"
			$URL:="keisuke-miyako/bge-small-en-v1.5-onnx"
			$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding")
			
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
			$options:={chat_template: $chat_template}
	End case 
	
	$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
	
End if 