Class extends _models

Class constructor($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $options : Object; $formula : 4D:C1709.Function; $event : cs:C1710.event.event)
	
	Super:C1705($port; $huggingfaces; $options; $formula; $event)
	
Function models() : cs:C1710.event.models
	
	var $models : Collection
	$models:=[]
	
	var $_model : Text
	For each ($_model; This:C1470._models)
		$models.push(cs:C1710.event.model.new($_model; True:C214))
	End for each 
	
	return cs:C1710.event.models.new($models)
	
Function start()
	
	var $ONNX : cs:C1710.workers.worker
	$ONNX:=cs:C1710.workers.worker.new(cs:C1710._server)
	$ONNX.start(This:C1470.options.port; This:C1470.options)
	
	If (This:C1470.event#Null:C1517) && (OB Instance of:C1731(This:C1470.event; cs:C1710.event.event))
		This:C1470.event.onSuccess.call(This:C1470; This:C1470.options; This:C1470.models())
	End if 
	