Class extends _download

property URL : Text

Class constructor($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $options : Object; $formula : 4D:C1709.Function; $event : cs:C1710.event.event)
	
	Super:C1705($options; $formula; $event)
	
	
	
	
	This:C1470.file:=$file
	This:C1470.URL:=$URL
	
	This:C1470.options.embeddings:=True:C214  //not embedding
	This:C1470.options.port:=$port
	This:C1470.options.model:=$file
	
	Case of 
		: (OB Instance of:C1731($file; 4D:C1709.File))
			If (Not:C34(This:C1470.file.exists))
				If (This:C1470.file.parent#Null:C1517)
					This:C1470.file.parent.create()
					This:C1470.head()
				End if 
			Else 
				This:C1470.start()
			End if 
		Else 
			//hugging face mode
			This:C1470.options.model:=This:C1470.URL
			This:C1470.file:={name: This:C1470.URL}
			This:C1470.start()
	End case 
	
Function models() : cs:C1710.event.models
	
	var $model : cs:C1710.event.model
	$model:=cs:C1710.event.model.new(This:C1470.options.model.name; Not:C34(This:C1470.options.model.exists))
	var $models : cs:C1710.event.models
	return cs:C1710.event.models.new([$model])
	
Function start()
	
	var $onnx : cs:C1710.workers.worker
	$onnx:=cs:C1710.workers.worker.new(cs:C1710._server)
	$onnx.start(This:C1470.options.port; This:C1470.options)
	
	If (This:C1470.event#Null:C1517) && (OB Instance of:C1731(This:C1470.event; cs:C1710.event.event))
		This:C1470.event.onSuccess.call(This:C1470; This:C1470.options; This:C1470.models())
	End if 
	
Function onResponse($request : 4D:C1709.HTTPRequest; $event : Object)
	
	If ($request.dataType="blob") && ($request.response.body#Null:C1517)
		This:C1470._fileHandle.writeBlob($request.response.body)
	End if 
	
	Case of 
		: (This:C1470.range.end=0)  //simple get
			If ($request.response.status=200)
				This:C1470._fileHandle:=Null:C1517
				If (This:C1470.event#Null:C1517) && (OB Instance of:C1731(This:C1470.event; cs:C1710.event.event))
					This:C1470.event.onResponse.call(This:C1470; $request; $event)
				End if 
				This:C1470.start()
			End if 
		Else   //range get
			If ([200; 206].includes($request.response.status))
				This:C1470.range.start:=This:C1470._fileHandle.getSize()
				If (This:C1470.range.start<This:C1470.range.length)
					var $end; $length : Real
					$end:=This:C1470.range.start+(This:C1470.bufferSize-1)
					$length:=This:C1470.range.length-1
					This:C1470.range.end:=$end>=$length ? $length : $end
					This:C1470.headers.Range:="bytes="+String:C10(This:C1470.range.start)+"-"+String:C10(This:C1470.range.end)
					4D:C1709.HTTPRequest.new(This:C1470.URL; This:C1470)
				Else 
					This:C1470._fileHandle:=Null:C1517
					If (This:C1470.event#Null:C1517) && (OB Instance of:C1731(This:C1470.event; cs:C1710.event.event))
						This:C1470.event.onResponse.call(This:C1470; $request; $event)
					End if 
					This:C1470.start()
				End if 
			End if 
			
	End case 