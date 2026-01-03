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
	
	If (Value type:C1509($option.embeggings_model)=Is object:K8:27)\
		 && (OB Instance of:C1731($option.embeggings_model; 4D:C1709.Folder))\
		 && ($option.embeggings_model.exists)
		$command+=" -e "
		$command+=This:C1470.escape(This:C1470.expand($option.embeggings_model).file("model.onnx").path)
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
	
	var $arg : Object
	var $valueType : Integer
	var $key : Text
	
	For each ($arg; OB Entries:C1720($option))
		Case of 
			: (["m"; "e"; "h"; "p"; "i"; "o"; "port"].includes($arg.key))
				continue
		End case 
		$valueType:=Value type:C1509($arg.value)
		$key:=Replace string:C233($arg.key; "_"; "-"; *)
		Case of 
			: ($valueType=Is real:K8:4)
				$command+=(" --"+$key+" "+String:C10($arg.value)+" ")
			: ($valueType=Is text:K8:3)
				$command+=(" --"+$key+" "+This:C1470.escape($arg.value)+" ")
			: ($valueType=Is boolean:K8:9) && ($arg.value)
				$command+=(" --"+$key+" ")
			: ($valueType=Is object:K8:27) && (OB Instance of:C1731($arg.value; 4D:C1709.File))
				$command+=(" --"+$key+" "+This:C1470.escape(This:C1470.expand($arg.value).path))
			Else 
				//
		End case 
	End for each 
	
	//SET TEXT TO PASTEBOARD($command)
	
	ALERT:C41($command)
	
	return This:C1470.controller.execute($command).worker