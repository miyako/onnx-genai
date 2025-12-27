Class extends _interface

Class constructor($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $options : Object; $event : cs:C1710.event.event)
	
	Super:C1705()
	
	var $ONNX : cs:C1710.workers.worker
	$ONNX:=cs:C1710.workers.worker.new(cs:C1710._server)
	
	If (Not:C34($ONNX.isRunning($port)))
		
		var $homeFolder : 4D:C1709.Folder
		$homeFolder:=Folder:C1567(fk home folder:K87:24).folder(".ONNX")
		
		If ($port=0) || ($port<0) || ($port>65535)
			$port:=8080
		End if 
		
		This:C1470._main($port; $huggingfaces; $options; $event)
		
	End if 
	
Function _main($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $options : Object; $event : cs:C1710.event.event)
	
	main({port: $port; huggingfaces: $huggingfaces; options: $options; event: $event}; This:C1470._onTCP)