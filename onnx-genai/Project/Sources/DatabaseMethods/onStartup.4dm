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
		: (False:C215)
			$chat_template:="{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '<|system|>\\n' + message['content'] + '<|end|>\\n' -}}\n    {%- elif message['role'] == 'user' -%}\n        {{- '<|user|>\\n' + message['content'] + '<|end|>\\n' -}}\n    "+"{%- elif message['role'] == 'assistant' -%}\n        {{- '<|assistant|>\\n' + message['content'] + '<|end|>\\n' -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- '<|assistant|>\\n' -}}\n{%- endif -%}\n"
			$chat_template:="{%- for message in messages -%}\n{%- if message.role == \"system\" -%}\n<|system|>\n{{ message.content }}\n<|end|>\n{%- elif message.role == \"user\" -%}\n<|user|>\n{{ message.content }}\n<|end|>\n{%- elif message.role == \"assistant\" -%}\n<|assistant|>\n{{ message.c"+"ontent }}\n<|end|>\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n<|assistant|>\n{%- endif -%}\n"
			$folder:=$homeFolder.folder("SmolLM2-1.7B-onnx-int4-cpu")
			$path:="keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu"
			$URL:="keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:="{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + 1) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'assistant'"+" %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{'<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}\n"
			$folder:=$homeFolder.folder("gemma-2-2B-it-onnx-int4-cpu")
			$path:="keisuke-miyako/gemma-2-2B-it-onnx-int4-cpu"
			$URL:="keisuke-miyako/gemma-2-2B-it-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:="{% for message in messages %}\n    {% if message['role'] == 'system' %}\n        {{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}\n    {% elif message['role'] == 'user' %}\n        {{ '<|im_start|>user\\n' + message['content'] + '<|im_end"+"|>\\n<|im_start|>assistant\\n' }}\n        {% if enable_thinking | default(true) %}\n            {{ '<|thought|>\\n' }}\n        {% endif %}\n    {% elif message['role'] == 'assistant' %}\n        {{ message['content'] + '<|im_end|>\\n' }}\n    {% endif %}\n{% e"+"ndfor %}\n"
			$folder:=$homeFolder.folder("Qwen3-1.7B-onnx-int4-cpu")
			$path:="keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:="{% for message in messages %}\n    {% if loop.first and message['role'] == 'system' %}\n        {{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}\n    {% elif message['role'] == 'user' %}\n        {{ '<|im_start|>user\\n' + message['conten"+"t'] + '<|im_end|>\\n' }}\n    {% elif message['role'] == 'assistant' %}\n        {{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}\n    {% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n    {{ '<|im_start|>assistant\\n' }}\n{% endi"+"f %}\n"
			$folder:=$homeFolder.folder("Qwen2.5-1.5B-onnx-int4-cpu")
			$path:="keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:="{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- message['content'] + '\\n' -%}{%- elif message['role'] == 'user' -%}{{- '[INST] ' + message['content'] + ' [/INST]' -%}{%- elif message['role'] == 'assistant' -%}{{- message['conte"+"nt'] + '\\n' -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}{{- '\\n' -%}{%- endif -%}\n"
			$folder:=$homeFolder.folder("Baguettotron-onnx-int4-cpu")
			$path:="keisuke-miyako/Baguettotron-onnx-int4-cpu"
			$URL:="keisuke-miyako/Baguettotron-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:="{% for message in messages %}\n    {{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}\n{% endfor %}\n{% if add_generation_prompt %}\n    {{ '<|im_start|>assistant\\n' }}\n{% endif %}"
			$folder:=$homeFolder.folder("EuroLLM-1.7B-Instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (True:C214)
			$chat_template:="<|begin_of_text|>{% for message in messages %}\n{% if message['role'] == 'system' %}\n<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'user' %}\n<|start_header_id|>user<|end_header_id|>\n\n{{ messag"+"e['content'] }}<|eot_id|>\n{% elif message['role'] == 'assistant' %}\n<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<|start_header_id|>assistant<|end_header_id|>"+"\n\n{% endif %}"
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