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
		: (False:C215)  // Granite3
			$chat_template:="{%- if tools %}\n    {{- '<|start_of_role|>available_tools<|end_of_role|>\n' }}\n    {%- for tool in tools %}\n    {{- tool | tojson(indent=4) }}\n    {%- if not loop.last %}\n        {{- '\n\n' }}\n    {%- endif %}\n    {%- endfor %}\n    {{- '<|end_of_text|>\n'"+" }}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n    {{- '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'user' %}\n    {{- '<|start_of_role|"+">user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>'  + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assist"+"ant_tool_call' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'tool_response' %}\n    {{- '<|start_of_role|>tool_response<|end_of_role|>' + message['conte"+"nt'] + '<|end_of_text|>\n' }}\n    {%- endif %}\n    {%- if loop.last and add_generation_prompt %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>' }}\n    {%- endif %}\n{%- endfor %}"
			$folder:=$homeFolder.folder("granite-3.0-2b-instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/granite-3.0-2b-instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/granite-3.0-2b-instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //Gemma2
			$chat_template:="{{ bos_token }}\n{% for message in messages %}\n    {% if message['role'] == 'user' %}\n        {{ '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}\n    {% elif message['role'] == 'assistant' %}\n        {{ '<start_of_turn>model\\n"+"' + message['content'] | trim + '<end_of_turn>\\n' }}\n    {% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n    {{ '<start_of_turn>model\\n' }}\n{% endif %}"
			$folder:=$homeFolder.folder("gemma-2-2b-it-onnx-int4-cpu")
			$path:="keisuke-miyako/gemma-2-2b-it-onnx-int4-cpu"
			$URL:="keisuke-miyako/gemma-2-2b-it-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //InternLM2 (Do not use this)
			$chat_template:="{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
			$folder:=$homeFolder.folder("internlm2_5-1_8b-chat-onnx-in4-cpu")
			$path:="keisuke-miyako/internlm2_5-1_8b-chat-onnx-in4-cpu"
			$URL:="keisuke-miyako/internlm2_5-1_8b-chat-onnx-in4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //EXAONE (Do not use this)
			$chat_template:="{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ '[|system|][|endofturn|]\n' }}{% endif %}{{ '[|' + message['role'] + '|]' + message['content'] }}{% if message['role'] == 'user' %}{{ '\n' }}{% else %}{{ '[|endofturn|]\n'"+" }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '[|assistant|]' }}{% endif %}"
			$folder:=$homeFolder.folder("EXAONE-3.5-2.4B-Instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/EXAONE-3.5-2.4B-Instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/EXAONE-3.5-2.4B-Instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //danube2
			$chat_template:="<|begin_of_text|>{% for message in messages %}\n{% if message['role'] == 'system' %}\n<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'user' %}\n<|start_header_id|>user<|end_header_id|>\n\n{{ messag"+"e['content'] }}<|eot_id|>\n{% elif message['role'] == 'assistant' %}\n<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<|start_header_id|>assistant<|end_header_id|>"+"\n\n{% endif %}"
			$folder:=$homeFolder.folder("h2o-danube2-1.8b-chat-onnx-int4-cpu")
			$path:="keisuke-miyako/h2o-danube2-1.8b-chat-onnx-int4-cpu"
			$URL:="keisuke-miyako/h2o-danube2-1.8b-chat-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //danube
			$chat_template:="<|begin_of_text|>{% for message in messages %}\n{% if message['role'] == 'system' %}\n<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'user' %}\n<|start_header_id|>user<|end_header_id|>\n\n{{ messag"+"e['content'] }}<|eot_id|>\n{% elif message['role'] == 'assistant' %}\n<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<|start_header_id|>assistant<|end_header_id|>"+"\n\n{% endif %}"
			$folder:=$homeFolder.folder("h2o-danube-1.8b-chat-onnx-int4-cpu")
			$path:="keisuke-miyako/h2o-danube-1.8b-chat-onnx-int4-cpu"
			$URL:="keisuke-miyako/h2o-danube-1.8b-chat-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //Phi 4 mini (official)
			$chat_template:="{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\\n' + message['content'] + '<|end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' + '\\n' }}{% endif %}"
			$folder:=$homeFolder.folder("Phi-4-mini-instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/Phi-4-mini-instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/Phi-4-mini-instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //Phi 3.5 mini (official)
			$chat_template:="{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\\n' + message['content'] + '<|end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' + '\\n' }}{% endif %}"
			$folder:=$homeFolder.folder("Phi-3.5-mini-instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/Phi-3.5-mini-instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/Phi-3.5-mini-instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //SmolLM2
			$chat_template:="{%- for message in messages -%}\n{%- if message.role == \"system\" -%}\n<|system|>\n{{ message.content }}\n<|end|>\n{%- elif message.role == \"user\" -%}\n<|user|>\n{{ message.content }}\n<|end|>\n{%- elif message.role == \"assistant\" -%}\n<|assistant|>\n{{ message.c"+"ontent }}\n<|end|>\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n<|assistant|>\n{%- endif -%}\n"
			$folder:=$homeFolder.folder("SmolLM2-1.7B-onnx-int4-cpu")
			$path:="keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu"
			$URL:="keisuke-miyako/SmolLM2-1.7B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //Qwen3 
			$chat_template:="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are "+"provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}{{ tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function nam"+"e and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\"+"n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n\n{%- set ns = namespace(last_query_index=messages|length - 1) %}\n{%- for message in messages %}\n    {%- if loop.first and message.role == 'system' %}{% continue %}{% endif %}\n "+"   \n    {%- set content = message.content if message.content is not none else \"\" %}\n    \n    {{- '<|im_start|>' + message.role + '\\n' }}\n    {%- if message.role == 'assistant' and '<think>' in content %}\n        {%- if loop.index0 < ns.last_query_inde"+"x %}\n            {# Pruning logic: Only keep the most recent thinking block #}\n            {{- content.split('</think>')[-1] }}\n        {%- else %}\n            {{- content }}\n        {%- endif %}\n    {%- else %}\n        {{- content }}\n    {%- endif %}"+"\n    {{- '<|im_end|>\\n' }}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking %}\n        {{- '<think>\\n' }}\n    {%- elif enable_thinking == false %}\n        {{- '<think>\\n</think>' }}\n    {%-"+" endif %}\n{%- endif %}"
			$folder:=$homeFolder.folder("Qwen3-1.7B-onnx-int4-cpu")
			$path:="keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Qwen3-1.7B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //Qwen2.5
			$chat_template:="{% for message in messages %}\n    {% if loop.first and message['role'] != 'system' %}\n        {{ '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {% endif %}\n    {{ '<|im_start|>' + message["+"'role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}\n{% endfor %}\n{% if add_generation_prompt %}\n    {{ '<|im_start|>assistant\\n' }}\n{% endif %}"
			$folder:=$homeFolder.folder("Qwen2.5-1.5B-onnx-int4-cpu")
			$path:="keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu"
			$URL:="keisuke-miyako/Qwen2.5-1.5B-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //Baguettotron
			$chat_template:="{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- message['content'] + '\\n' -%}{%- elif message['role'] == 'user' -%}{{- '[INST] ' + message['content'] + ' [/INST]' -%}{%- elif message['role'] == 'assistant' -%}{{- message['conte"+"nt'] + '\\n' -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}{{- '\\n' -%}{%- endif -%}\n"
			$folder:=$homeFolder.folder("Baguettotron-onnx-int4-cpu")
			$path:="keisuke-miyako/Baguettotron-onnx-int4-cpu"
			$URL:="keisuke-miyako/Baguettotron-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)  //EuroLLM
			$chat_template:="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}<|im_start|>system\n{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% endif %}<|im_end|>\n{% for message in messages %}{% if message['rol"+"e'] != 'system' %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
			$folder:=$homeFolder.folder("EuroLLM-1.7B-Instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/EuroLLM-1.7B-Instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat])
			$options:={chat_template: $chat_template}
		: (False:C215)
			$chat_template:="<|begin_of_text|>{% for message in messages %}\n{% if message['role'] == 'system' %}\n<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% elif message['role'] == 'user' %}\n<|start_header_id|>user<|end_header_id|>\n\n{{ messag"+"e['content'] }}<|eot_id|>\n{% elif message['role'] == 'assistant' %}\n<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<|start_header_id|>assistant<|end_header_id|>"+"\n\n{% endif %}"
			$folder:=$homeFolder.folder("Llama-3.2-1B-Instruct-onnx-int4-cpu")
			$path:="keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu"
			$URL:="keisuke-miyako/Llama-3.2-1B-Instruct-onnx-int4-cpu"
			$chat:=cs:C1710.event.huggingface.new($folder; $URL; $path; "chat.completion")
			
			//$folder:=$homeFolder.folder("bge-small-en-v1.5-onnx")
			//$path:="keisuke-miyako/bge-small-en-v1.5-onnx"
			//$URL:="keisuke-miyako/bge-small-en-v1.5-onnx"
			//$embeddings:=cs.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			
			//$folder:=$homeFolder.folder("nomic-embed-text-v1.5-onnx")
			//$path:="keisuke-miyako/nomic-embed-text-v1.5-onnx"
			//$URL:="keisuke-miyako/nomic-embed-text-v1.5-onnx"
			//$embeddings:=cs.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			
			$folder:=$homeFolder.folder("embeddinggemma-300m-onnx")
			$path:="keisuke-miyako/embeddinggemma-300m-onnx"
			$URL:="keisuke-miyako/embeddinggemma-300m-onnx"
			$embeddings:=cs:C1710.event.huggingface.new($folder; $URL; $path; "embedding"; "model_quantized.onnx")
			
			$huggingfaces:=cs:C1710.event.huggingfaces.new([$chat; $embeddings])
			$options:={chat_template: $chat_template}
	End case 
	
	$ONNX:=cs:C1710.ONNX.new($port; $huggingfaces; $homeFolder; $options; $event)
	
End if 