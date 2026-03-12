Class extends _ONNX

Class constructor($controller : 4D:C1709.Class)
	
	Super:C1705("ONNX-genai"; $controller)
	
Function start($option : Object) : 4D:C1709.SystemWorker
	
	This:C1470.bind($option; ["onTerminate"])
	
	var $command : Text
	$command:=This:C1470.escape(This:C1470.executablePath)
	
	$command+=" -s "
	
	If (Value type:C1509($option.chat_completion_model)=Is object:K8:27)\
		 && (OB Instance of:C1731($option.chat_completion_model; 4D:C1709.Folder))\
		 && ($option.chat_completion_model.exists)
		$command+=" -m "
		$command+=This:C1470.escape(This:C1470.expand($option.chat_completion_model).path)
		$command+=" "
	End if 
	
	If (Value type:C1509($option.embeddings_model)=Is object:K8:27)\
		 && (OB Instance of:C1731($option.embeddings_model; 4D:C1709.Folder))\
		 && ($option.embeddings_model.exists)
		$command+=" -e "
		$command+=This:C1470.escape(This:C1470.expand($option.embeddings_model).file($option.embeggings_model_name).path)
		$command+=" "
	End if 
	
	If (Value type:C1509($option.rerank_model)=Is object:K8:27)\
		 && (OB Instance of:C1731($option.rerank_model; 4D:C1709.Folder))\
		 && ($option.rerank_model.exists)
		$command+=" -r "
		$command+=This:C1470.escape(This:C1470.expand($option.rerank_model).file($option.rerank_model_name).path)
		$command+=" "
	End if 
	
	$command+=" -p "
	$command+=String:C10($option.port)
	$command+=" "
	
	If (Value type:C1509($option.host)=Is text:K8:3) && ($option.host#"")
		$command+=" -h "
		$command+=$option.host
		$command+=" "
	End if 
	
	var $chat_template : Text
	If (Value type:C1509($option.chat_template)=Is text:K8:3) && ($option.chat_template#"")
		$command+=" -t "
		$chat_template:=$option.chat_template
	End if 
	
	If (Value type:C1509($option.pooling)=Is text:K8:3) && ($option.pooling#"")
		Case of 
			: ($option.pooling="cls")
				$command+=" -c "
			: ($option.pooling="mean")
				//default
			: ($option.pooling="last-token")
				$command+=" -l "
			: ($option.pooling="multi-vector")  //ColBERT
				$command+=" -b "
			: ($option.pooling="splade")
				//not implemented
			: ($option.pooling="e2e")  //Universal Sentence Encoder
				$command+=" -d "
		End case 
	End if 
	
	var $arg : Object
	var $valueType : Integer
	var $key : Text
	
	For each ($arg; OB Entries:C1720($option))
		Case of 
			: (["i"; "o"; "s"; "j"; "c"; "l"; "b"; "d"; "_"; \
				"h"; "host"; \
				"p"; "port"; \
				"t"; "chat_template"; \
				"pooling"; \
				"r"; "rerank_model"; "rerank_model_name"; \
				"e"; "embeggings_model_name"; \
				"m"; "chat_completion_model"; "HF_TOKEN"].includes($arg.key))
				continue
		End case 
		$valueType:=Value type:C1509($arg.value)
		$key:=Replace string:C233($arg.key; "_"; "-"; *)
		$prefix:=Length:C16($key)=1 ? " -" : " --"
		
		Case of 
			: ($valueType=Is real:K8:4)
				$command+=($prefix+$key+" "+String:C10($arg.value)+" ")
			: ($valueType=Is text:K8:3)
				$command+=($prefix+$key+" "+This:C1470.escape($arg.value)+" ")
			: ($valueType=Is boolean:K8:9) && ($arg.value)
				$command+=($prefix+$key+" ")
			: ($valueType=Is object:K8:27) && (OB Instance of:C1731($arg.value; 4D:C1709.File))
				$command+=($prefix+$key+" "+This:C1470.escape(This:C1470.expand($arg.value).path))
			Else 
				//
		End case 
	End for each 
	
	//SET TEXT TO PASTEBOARD($command)
	
	return This:C1470.controller.execute($command; $chat_template#"" ? $chat_template : Null:C1517).worker