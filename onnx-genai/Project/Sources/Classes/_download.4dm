property that : cs:C1710._Model
property file : 4D:C1709.File
property folder : 4D:C1709.Folder
property URL : Text
property oid : Text
property onDownload : 4D:C1709.Function
property options : Object
property _onResponse : 4D:C1709.Function
property _fileHandle : 4D:C1709.FileHandle
property bufferSize : Integer
property range : Object
property method : Text
property dataType : Text
property automaticRedirections : Boolean
property returnResponseBody : Boolean
property decodeData : Boolean
property headers : Object
property event : cs:C1710.event.event
property agent : Object

Class constructor($that : cs:C1710._Model; \
$file : 4D:C1709.File; \
$folder : 4D:C1709.Folder; \
$oid : Text; \
$URL : Text; \
$options : Object; \
$formula : 4D:C1709.Function; \
$event : cs:C1710.event.event; \
$onDownload : 4D:C1709.Function)
	
	This:C1470.that:=$that
	This:C1470.file:=$file
	This:C1470.folder:=$folder
	This:C1470.URL:=$URL
	This:C1470.oid:=$oid
	This:C1470.onDownload:=$onDownload
	
	This:C1470.file.parent.create()
	
	This:C1470.method:="GET"
	This:C1470.dataType:="blob"
	This:C1470.automaticRedirections:=True:C214
	This:C1470.returnResponseBody:=False:C215
	This:C1470.decodeData:=False:C215
	This:C1470.headers:={Accept: "application/vnd.github+json"}
	
	This:C1470.bufferSize:=10*(1024^2)
	
	This:C1470._onResponse:=$formula
	This:C1470.event:=$event
	
	This:C1470.options:=$options#Null:C1517 ? $options : {}
	This:C1470.options.onTerminate:=This:C1470.event.onTerminate
	This:C1470.options.onStdErr:=This:C1470.event.onStdErr
	This:C1470.options.onStdOut:=This:C1470.event.onStdOut
	
	If (OB Instance of:C1731(4D:C1709.HTTPAgent; 4D:C1709.Class))
		This:C1470.agent:=4D:C1709.HTTPAgent.new({keepAlive: False:C215})
	End if 
	
Function head()
	
	This:C1470.method:="HEAD"
	This:C1470.range:={length: 0; start: 0; end: 0; ranges: False:C215}
	//HEAD; async onResponse not supported
	var $request : 4D:C1709.HTTPRequest
	$request:=4D:C1709.HTTPRequest.new(This:C1470.URL; This:C1470).wait()
	If ($request.response.status=200)
		This:C1470.method:="GET"
		If (Not:C34(This:C1470.decodeData))
			This:C1470.headers["Accept-Encoding"]:="identity"
		End if 
		If (Value type:C1509($request.response.headers["accept-ranges"])=Is text:K8:3) && \
			($request.response.headers["accept-ranges"]="bytes")
			This:C1470.range.length:=Num:C11($request.response.headers["content-length"])
		End if 
		This:C1470._fileHandle:=This:C1470.file.open("write")
		If (This:C1470.range.length#0)
			This:C1470.range.ranges:=True:C214
			var $end; $length : Real
			$end:=This:C1470.range.start+(This:C1470.bufferSize-1)
			$length:=This:C1470.range.length-1
			This:C1470.range.end:=$end>=$length ? $length : $end
			This:C1470.headers.Range:="bytes="+String:C10(This:C1470.range.start)+"-"+String:C10(This:C1470.range.end)
		End if 
		4D:C1709.HTTPRequest.new(This:C1470.URL; This:C1470)
	Else 
		This:C1470._onResponse.call(This:C1470; {success: False:C215}; This:C1470.options)
	End if 
	
Function onData($request : 4D:C1709.HTTPRequest; $event : Object)
	
	If (This:C1470._fileHandle#Null:C1517)
		If ($request.dataType="blob") && ($event.data#Null:C1517)
			This:C1470._fileHandle.writeBlob($event.data)
		End if 
		If (Not:C34(This:C1470.range.ranges))  //simple get
			This:C1470.range.end:=This:C1470._fileHandle.getSize()
			This:C1470.range.length:=Num:C11($request.response.headers["content-length"])
		End if 
		If (This:C1470.event#Null:C1517) && (OB Instance of:C1731(This:C1470.event; cs:C1710.event.event))
			This:C1470.event.onData.call(This:C1470; $request; $event)
		End if 
	End if 
	
Function onError($request : 4D:C1709.HTTPRequest; $event : Object)
	
	If (Value type:C1509(This:C1470._onResponse)=Is object:K8:27) && (OB Instance of:C1731(This:C1470._onResponse; 4D:C1709.Function))
		This:C1470._onResponse.call(This:C1470; {success: False:C215}; This:C1470.options)
		If (This:C1470._fileHandle#Null:C1517)
			This:C1470._fileHandle:=Null:C1517
			This:C1470.file.delete()
		End if 
	End if 
	
Function onResponse($request : 4D:C1709.HTTPRequest; $event : Object)
	
	If ($request.dataType="blob") && ($request.response.body#Null:C1517)
		This:C1470._fileHandle.writeBlob($request.response.body)
	End if 
	
	Case of 
		: (Not:C34(This:C1470.range.ranges))  //simple get
			If ($request.response.status=200)
				This:C1470._fileHandle:=Null:C1517
				If (This:C1470.event#Null:C1517) && (OB Instance of:C1731(This:C1470.event; cs:C1710.event.event))
					This:C1470.event.onResponse.call(This:C1470; $request; $event)
				End if 
				This:C1470.onDownload.call(This:C1470.that; This:C1470.oid)
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
					This:C1470.onDownload.call(This:C1470.that; This:C1470.oid)
				End if 
			End if 
	End case 
	
Function onTerminate($request : 4D:C1709.HTTPRequest; $event : Object)