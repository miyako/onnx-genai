Class extends _interface

Class constructor($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $HOME : 4D:C1709.Folder; $options : Object; $event : cs:C1710.event.event)
	
	Super:C1705()
	
	var $ONNX : cs:C1710.workers.worker
	$ONNX:=cs:C1710.workers.worker.new(cs:C1710._server)
	
	If (Not:C34($ONNX.isRunning($port)))
		
		If (Not:C34(OB Instance of:C1731($HOME; 4D:C1709.Folder))) || (Not:C34($HOME.exists))
			$HOME:=Folder:C1567(fk home folder:K87:24).folder(".ONNX")
		End if 
		
		Case of 
			: ($huggingfaces#Null:C1517) && (OB Instance of:C1731($huggingfaces; cs:C1710.event.huggingfaces)) && ($huggingfaces.huggingfaces.length#0)
			: (This:C1470._hasLocalModel($options))
				//local models only: nothing to download
				$huggingfaces:=cs:C1710.event.huggingfaces.new([])
			Else 
				$chat_template:="<|begin_of_text|>{% for message in messages %}\n{% if message['role'] == 'system' %}\n<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'user' %}\n<|start_header_id|>user<|end_header_id|>\n\n{{ messag"+"e['content'] }}<|eot_id|>\n{% elif message['role'] == 'assistant' %}\n<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<|start_header_id|>assistant<|end_header_id|>"+"\n\n{% endif %}"
				$folder:=$HOME.folder("Llama-3.2-1B-Instruct-onnx-int4-cpu")
				$path:="keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu"
				$URL:="keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu"
				$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			
				$folder:=$HOME.folder("embeddinggemma-300m-onnx")
				$path:="keisuke-miyako/embeddinggemma-300m-onnx"
				$URL:="keisuke-miyako/embeddinggemma-300m-onnx"
				$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			
				$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
				If ($options=Null:C1517)
					$options:={}
				End if 
				If ($options.chat_template=Null:C1517)
					$options.chat_template:=$chat_template
				End if 
		End case 
		
		If ($port=0) || ($port<0) || ($port>65535)
			$port:=8080
		End if 
		
		This:C1470._main($port; $huggingfaces; $HOME; $options; $event)
		
	End if 
	
Function _main($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $HOME : 4D:C1709.Folder; $options : Object; $event : cs:C1710.event.event)
	
	main({name: Split string:C1554(Current method name:C684; "."; sk trim spaces:K86:2).first(); port: $port; huggingfaces: $huggingfaces; HOME: $HOME; options: $options; event: $event}; This:C1470._onTCP)
	
Function _hasLocalModel($options : Object) : Boolean
	
	If ($options=Null:C1517)
		return False:C215
	End if 
	var $key : Text
	For each ($key; ["chat_completion_model"; "embeddings_model"; "rerank_model"; "decision_model"; "tts_model"])
		If (Value type:C1509($options[$key])=Is object:K8:27)
			If (OB Instance of:C1731($options[$key]; 4D:C1709.Folder))
				return True:C214
			End if 
		End if 
	End for each 
	return False:C215