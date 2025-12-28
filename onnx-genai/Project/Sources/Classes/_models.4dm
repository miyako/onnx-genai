property huggingfaces : cs:C1710.event.huggingfaces
property files : Collection
property _models : Collection
property options : Object
property event : cs:C1710.event.event
property _onResponse : 4D:C1709.Function

Class constructor($port : Integer; $huggingfaces : cs:C1710.event.huggingfaces; $options : Object; $formula : 4D:C1709.Function; $event : cs:C1710.event.event)
	
	This:C1470.huggingfaces:=$huggingfaces
	This:C1470.event:=$event
	This:C1470._onResponse:=$formula
	This:C1470.options:=$options#Null:C1517 ? $options : {}
	This:C1470.options.onTerminate:=This:C1470.event.onTerminate
	This:C1470.options.onStdErr:=This:C1470.event.onStdErr
	This:C1470.options.onStdOut:=This:C1470.event.onStdOut
	
	This:C1470.files:=[]
	This:C1470._models:=[]
	
	ARRAY LONGINT:C221($pos; 0)
	ARRAY LONGINT:C221($len; 0)
	
	var $URL : Text
	var $huggingface : cs:C1710.event.huggingface
	While (This:C1470.huggingfaces.huggingfaces.length#0)
		$huggingface:=This:C1470.huggingfaces.huggingfaces.shift()
		Case of 
			: (Match regex:C1019("https:\\/\\/huggingface\\.co\\/(.+)"; $huggingface.URL; 1; $pos; $len))
				$URL:=Substring:C12($huggingface.URL; $pos{1}; $len{1})
			: (Match regex:C1019("(^|\\/[^/]+){2,}"; $huggingface.URL; 1; $pos; $len))
				$URL:=$huggingface.URL
			Else 
				continue
		End case 
		var $components : Collection
		$components:=Split string:C1554($URL; "/")
		$BRANCH:="main"
		If ($components.length>1)
			$USER:=$components.shift()
			$REPO:=$components.shift()
			If ($components.length#0)
				$BRANCH:=$components.pop()
			End if 
			$URL:=["https://huggingface.co/api/models"; $USER; $REPO; "tree"; $BRANCH; "?recursive=true"].join("/")
			var $request : 4D:C1709.HTTPRequest
			$request:=4D:C1709.HTTPRequest.new($URL).wait()
			If ($request.response.status=200)
				If (Value type:C1509($request.response)=Is object:K8:27)
					$resources:=$request.response.body.map(Formula:C1597($1.result:={\
						USER: $2; \
						REPO: $3; \
						BRANCH: $4; \
						type: $1.value.type; \
						size: $1.value.size; \
						path: $1.value.path; \
						oid: $1.value.oid; \
						folder: $5}); $USER; $REPO; $BRANCH; $huggingface.folder)
					This:C1470.files:=This:C1470.files.combine($resources.query("type == :1"; "file"))
					This:C1470._models.push([$USER; $REPO].join("/"))
				End if 
			End if 
		End if 
	End while 
	
	For each ($file; This:C1470.files)
		cs:C1710._download.new(\
			This:C1470; \
			$file.folder.file($file.path); \
			$file.folder; \
			$file.oid; \
			["https://huggingface.co"; $file.USER; $file.REPO; "resolve"; $file.BRANCH; $file.path].join("/"); \
			This:C1470.options; \
			This:C1470._onResponse; \
			This:C1470.event; This:C1470.onDownload).head()
	End for each 
	
Function onDownload($oid : Text)
	
	This:C1470.files:=This:C1470.files.filter(Formula:C1597($1.value.oid#$2); $oid)
	
	If (This:C1470.files.length=0)
		This:C1470.start()
	End if 
	
Function models() : cs:C1710.event.models
	
	return cs:C1710.event.models.new()
	
Function start()