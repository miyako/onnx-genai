//
//  main.cpp
//  onnx-genai
//
//  Created by miyako on 2025/09/03.
//

#include "onnx-genai.h"

#ifdef WIN32
static std::string wchar_to_utf8(const wchar_t* wstr) {
    if (!wstr) return std::string();
    
    // Get required buffer size in bytes
    int size_needed = WideCharToMultiByte(
                                          CP_UTF8,            // convert to UTF-8
                                          0,                  // default flags
                                          wstr,               // source wide string
                                          -1,                 // null-terminated
                                          nullptr, 0,         // no output buffer yet
                                          nullptr, nullptr
                                          );
    
    if (size_needed <= 0) return std::string();
    
    // Allocate buffer
    std::string utf8str(size_needed, 0);
    
    // Perform conversion
    WideCharToMultiByte(
                        CP_UTF8,
                        0,
                        wstr,
                        -1,
                        &utf8str[0],
                        size_needed,
                        nullptr,
                        nullptr
                        );
    
    // Remove the extra null terminator added by WideCharToMultiByte
    if (!utf8str.empty() && utf8str.back() == '\0') {
        utf8str.pop_back();
    }
    
    return utf8str;
}
#endif

/*
float calculate_magnitude(const std::vector<float>& vec) {
    // Helper to calculate vector magnitude for L2 Normalization
    float sum_squares = 0.0f;
    for (float val : vec) sum_squares += val * val;
    return std::sqrt(sum_squares);
}
*/

static bool GetMaxSeqLen(Ort::Session* session, size_t *max_seq_len) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session->GetInputNameAllocated(0, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    // shape = [batch, seq_len]
    if (shape.size() != 2 || shape[1] <= 0) {
        return false;
    }else{
        *max_seq_len = static_cast<size_t>(shape[1]);
    }
    return true;
}

#pragma mark -

static void usage(void)
{
    fprintf(stderr, "Usage:  onnx-genai -m model -i input\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c path     : %s\n", 'm' , "model");
    fprintf(stderr, " -%c path     : %s\n", 'e' , "embedding model");
    //
    fprintf(stderr, " -%c path     : %s\n", 'i' , "input");
    fprintf(stderr, " %c           : %s\n", '-' , "use stdin for input");
    fprintf(stderr, " -%c path     : %s\n", 'o' , "output (default=stdout)");
    //
    exit(1);
}

extern OPTARG_T optarg;
extern int optind, opterr, optopt;

#ifdef WIN32
OPTARG_T optarg = 0;
int opterr = 1;
int optind = 1;
int optopt = 0;
int getopt(int argc, OPTARG_T *argv, OPTARG_T opts) {
    
    static int sp = 1;
    register int c;
    register OPTARG_T cp;
    
    if(sp == 1)
        if(optind >= argc ||
           argv[optind][0] != '-' || argv[optind][1] == '\0')
            return(EOF);
        else if(wcscmp(argv[optind], L"--") == NULL) {
            optind++;
            return(EOF);
        }
    optopt = c = argv[optind][sp];
    if(c == ':' || (cp=wcschr(opts, c)) == NULL) {
        ERR(L": illegal option -- ", c);
        if(argv[optind][++sp] == '\0') {
            optind++;
            sp = 1;
        }
        return('?');
    }
    if(*++cp == ':') {
        if(argv[optind][sp+1] != '\0')
            optarg = &argv[optind++][sp+1];
        else if(++optind >= argc) {
            ERR(L": option requires an argument -- ", c);
            sp = 1;
            return('?');
        } else
            optarg = argv[optind++];
        sp = 1;
    } else {
        if(argv[optind][++sp] == '\0') {
            sp = 1;
            optind++;
        }
        optarg = NULL;
    }
    return(c);
}
#define ARGS (OPTARG_T)L"m:e:i:o:sp:-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:e:i:o:sp:-h"
#define _atoi atoi
#define _atof atof
#endif

#pragma mark -

static long long get_created_timestamp() {
    // std::time(nullptr) returns the current time as a time_t (seconds since epoch)
    return static_cast<long long>(std::time(nullptr));
}

namespace fs = std::filesystem;
static std::string get_model_name(std::string model_path) {
    // 1. Create a path object
    fs::path path(model_path);
    
    // 2. Handle trailing slashes (e.g., "models/phi-3/")
    // If the path ends in a separator, filename() might return empty.
    if (path.filename().empty()) {
        path = path.parent_path();
    }
    
    // 3. Return the folder/filename
    // .filename() returns "phi-3.onnx" (with extension)
    // .stem() returns "phi-3" (removes extension)
    return path.stem().string();
}

// Generate a fingerprint based on model identity and hardware
static std::string get_system_fingerprint(const std::string& model_path, const std::string& provider) {
    // 1. Combine identifying factors (Model + Engine)
    std::string identifier = model_path + "_" + provider;
    
    // 2. Hash the string to get a unique number
    std::hash<std::string> hasher;
    size_t hash = hasher(identifier);
    
    // 3. Format as hex (e.g., "fp_1a2b3c4d")
    std::stringstream ss;
    ss << "fp_" << std::hex << hash;
    
    return ss.str();
}

static std::string generate_uuid_v4() {
    std::stringstream ss;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::uniform_int_distribution<> dis2(8, 11);
    
    ss << std::hex;
    for (int i = 0; i < 8; i++) { ss << dis(gen); }
    ss << "-";
    for (int i = 0; i < 4; i++) { ss << dis(gen); }
    ss << "-4"; // UUID version 4
    for (int i = 0; i < 3; i++) { ss << dis(gen); }
    ss << "-";
    ss << dis2(gen); // UUID variant
    for (int i = 0; i < 3; i++) { ss << dis(gen); }
    ss << "-";
    for (int i = 0; i < 12; i++) { ss << dis(gen); }
    
    return ss.str();
}

static std::string get_openai_id() {
    return "chatcmpl-" + generate_uuid_v4();
}

static std::string generate_openai_style_id() {
    const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    
    std::string id = "chatcmpl-";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, max_index - 1);
    
    for (int i = 0; i < 29; ++i) {
        id += charset[dis(gen)];
    }
    return id;
}

static void parse_request(
                          const std::string &json,
                          std::string &prompt,
                          unsigned int *max_tokens,
                          unsigned int *top_k,
                          double *top_p,
                          double *temperature,
                          unsigned int *n,
                          bool *is_stream) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    Json::CharReader *reader = builder.newCharReader();
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    delete reader;
    
    if(parse)
    {
        if(root.isObject())
        {
            Json::Value messages_node = root["messages"];
            if(messages_node.isArray())
            {
                prompt = "";
                for(Json::Value::const_iterator it = messages_node.begin() ; it != messages_node.end() ; it++)
                {
                    if(it->isObject())
                    {
                        Json::Value defaultValue = "";
                        Json::Value role = it->get("role", defaultValue);
                        Json::Value content = it->get("content", defaultValue);
                        if ((role.isString()) && (content.isString()))
                        {
                            std::string role_str = role.asString();
                            std::string content_str = content.asString();
                            if ((role_str.length() != 0) && (content_str.length() != 0))
                            {
                                prompt += "<|";
                                prompt += role_str;
                                prompt += "|>";
                                prompt += content_str;
                                prompt += "<|end|>";
                            }
                        }
                    }
                }
                if(prompt.length() != 0) prompt += "<|assistant|>";
            }
            
            Json::Value top_p_node = root["top_p"];
            if(top_p_node.isNumeric())
            {
                *top_p = top_p_node.asDouble();
            }
            Json::Value top_k_node = root["top_k"];
            if(top_k_node.isNumeric())
            {
                *top_k = top_k_node.asInt();
            }
            Json::Value max_tokens_node = root["max_tokens"];
            if(max_tokens_node.isNumeric())
            {
                *max_tokens = max_tokens_node.asInt();
            }
            /*
             only these are set by AI-Kit
             */
            Json::Value temperature_node = root["temperature"];
            if(temperature_node.isNumeric())
            {
                *temperature = temperature_node.asDouble();
            }
            Json::Value n_node = root["n"];
            if(n_node.isNumeric())
            {
                *n = n_node.asInt();
            }
            max_tokens_node = root["max_completion_tokens"];
            if(max_tokens_node.isNumeric())
            {
                *max_tokens = max_tokens_node.asInt();
            }
            Json::Value stream_node = root["stream"];
            if(stream_node.isBool())
            {
                *is_stream = stream_node.asBool();
            }
        }
    }
}

static std::string run_embedding(
                                 Ort::Session* session,
                                 OgaTokenizer* tokenizer,
                                 const std::string& modelName,
                                 const std::string& request_body
                                 ) {
    std::string input_text;
    
    // 1. Parse JSON Request
    // We expect { "model": "...", "input": "text" }
    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errs;
    std::istringstream s(request_body);
    if (!Json::parseFromStream(reader, s, &root, &errs)) {
        throw std::runtime_error("Invalid JSON body");
    }
    
    // Handle "input": supports string or array of strings (OpenAI spec)
    // For simplicity here, we handle a single string.
    if (root["input"].isString()) {
        input_text = root["input"].asString();
    } else if (root["input"].isArray() && root["input"].size() > 0) {
        // Just take the first one for this example implementation
        input_text = root["input"][0].asString();
    } else {
        throw std::runtime_error("Invalid 'input' field. String expected.");
    }
    
    // 2. Tokenize (Using OgaTokenizer)
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(input_text.c_str(), *sequences);
    size_t seq_len = sequences->SequenceCount(0);
    const int32_t* token_data = sequences->SequenceData(0);
    
    // 3. Prepare Inputs for ONNX Runtime
    // Most BERT/Encoder models expect int64 for input_ids
    std::vector<int64_t> input_ids(seq_len);
    std::vector<int64_t> attention_mask(seq_len);
    
    for (size_t i = 0; i < seq_len; i++) {
        input_ids[i] = static_cast<int64_t>(token_data[i]);
        attention_mask[i] = 1; // All tokens attended to
    }
    
    // Create Ort Tensors
    
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, (int64_t)seq_len}; // Batch size 1
    
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                                                                    memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
                                                                         memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());
    
    // 4. Run Inference
    // Standard names for transformer models. Check your specific ONNX model inputs via Netron.
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"last_hidden_state"}; // Or "embeddings", check your model
    
    Ort::Value inputs[] = { std::move(input_ids_tensor), std::move(attention_mask_tensor) };
    
    auto output_tensors = session->Run(
                                       Ort::RunOptions{nullptr},
                                       input_names,
                                       inputs,
                                       2,
                                       output_names,
                                       1
                                       );
    
    // 5. Post-Process (Mean Pooling & Normalization)
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape(); // [Batch, SeqLen, HiddenSize]
    
    int64_t hidden_size = shape[2];
    std::vector<float> embedding(hidden_size, 0.0f);
    
    // Mean Pooling: Sum vectors across sequence length
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t h = 0; h < hidden_size; h++) {
            embedding[h] += floatarr[i * hidden_size + h];
        }
    }
    
    // Divide by sequence length and calculate Norm
    double norm = 0.0;
    for (size_t h = 0; h < hidden_size; h++) {
        embedding[h] /= (float)seq_len;
        norm += embedding[h] * embedding[h];
    }
    
    // L2 Normalize (OpenAI embeddings are unit length)
    norm = std::sqrt(norm);
    // Avoid division by zero
    if (norm > 1e-12) {
        for (size_t h = 0; h < hidden_size; h++) {
            embedding[h] /= (float)norm;
        }
    }
    
    // 6. Build JSON Response
    Json::Value rootNode(Json::objectValue);
    
    rootNode["object"] = "list";
    rootNode["model"] = modelName;
    
    Json::Value dataArray(Json::arrayValue);
    Json::Value dataItem(Json::objectValue);
    
    dataItem["object"] = "embedding";
    dataItem["index"] = 0;
    
    Json::Value embeddingArray(Json::arrayValue);
    for(float val : embedding) {
        embeddingArray.append(val);
    }
    dataItem["embedding"] = embeddingArray;
    
    dataArray.append(dataItem);
    rootNode["data"] = dataArray;
    
    Json::Value usageNode(Json::objectValue);
    usageNode["prompt_tokens"] = (Json::Int)seq_len;
    usageNode["total_tokens"] = (Json::Int)seq_len;
    rootNode["usage"] = usageNode;
    
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

static void before_run_inference(
                                 const std::string& request_body,
                                 std::string &prompt,
                                 unsigned int *max_tokens,
                                 unsigned int *top_k,
                                 double *top_p,
                                 double *temperature,
                                 unsigned int *n,
                                 bool *is_stream) {
    
    parse_request(request_body, prompt, max_tokens, top_k, top_p, temperature, n, is_stream);
}

#pragma mark -

static std::string run_inference(
                                 OgaModel* model,
                                 OgaTokenizer* tokenizer,
                                 const std::string& modelName,
                                 const std::string& fingerprint,
                                 long long created,
                                 unsigned int max_tokens,
                                 unsigned int top_k,
                                 double top_p,
                                 double temperature,
                                 unsigned int n,
                                 std::string prompt
                                 ) {
    /*
     The chat completion object
     https://platform.openai.com/docs/api-reference/chat/object
     */
    std::string content;
    Json::Value rootNode(Json::objectValue);
    size_t completion_tokens = 0;
    size_t input_token_count = 0;
    std::string finish_reason = "stop";//length, content_filter, tool_calls, function_call
    
    try {
        // Create Tokenizer Stream
        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
        
        // Encode Prompt
        auto input_sequences = OgaSequences::Create();
        tokenizer->Encode(prompt.c_str(), *input_sequences);
        input_token_count = input_sequences->SequenceCount(0);
        double max_length = (double)(input_token_count + max_tokens);
        
        // Set Generation Parameters
        auto params = OgaGeneratorParams::Create(*model);
        params->SetSearchOption("max_length", max_length);
        params->SetSearchOption("top_k", top_k);
        params->SetSearchOption("top_p", top_p);
        params->SetSearchOption("temperature", temperature);
        params->SetSearchOption("num_return_sequences", n);
        
        // Create Generator
        // Generator is stateful; we need 1 per request.
        auto generator = OgaGenerator::Create(*model, *params);
        generator->AppendTokenSequences(*input_sequences);
                
        // Create a vector of streams
        // Decoding is stateful; we need 1 decoder per sequence.
        std::vector<std::string> generated_responses(n);
        std::vector<std::unique_ptr<OgaTokenizerStream>> streams;
        for (int i = 0; i < n; i++) {
            streams.push_back(OgaTokenizerStream::Create(*tokenizer));
        }
        
        // Start Generating
        while (1) {
            generator->GenerateNextToken();
            if(generator->IsDone()) break;
            // Iterate through each sequence (0 to n-1) to collect results
            for (int i = 0; i < n; i++) {
                // Get the full sequence data for the i-th choice
                const auto* seq_data = generator->GetSequenceData(i);
                size_t seq_len = generator->GetSequenceCount(i);
                // Safety check to ensure we have data
                if (seq_len == 0) continue;
                // Get the most recently generated token
                int32_t new_token = seq_data[seq_len - 1];
                // Decode using the specific stream for this sequence
                const char* token_str = streams[i]->Decode(new_token);
                if (token_str) {
                    generated_responses[i] += token_str;
                    completion_tokens++;
                }
            }
        }
        
        Json::Int total_tokens = (Json::Int)(input_token_count + completion_tokens);
        if (total_tokens >= max_length) {
            finish_reason = "length";
        }
        // Build Response JSON
        rootNode["id"] = generate_openai_style_id();
        rootNode["object"] = "chat.completion";
        rootNode["created"] = created;
        rootNode["model"] = modelName;
        rootNode["system_fingerprint"] = fingerprint;//Deprecated
        rootNode["service_tier"] = "default";
        Json::Value choicesNode(Json::arrayValue);
        
        for (int i = 0; i < n; i++) {
            Json::Value choiceNode(Json::objectValue);
            choiceNode["index"] = i;
            Json::Value messageNode(Json::objectValue);
            messageNode["role"] = "assistant";
            messageNode["content"] = generated_responses[i].c_str();
            messageNode["refusal"] = Json::nullValue;
            choiceNode["message"] = messageNode;
            choicesNode.append(choiceNode);
            choiceNode["logprobs"] = Json::nullValue;
            choiceNode["finish_reason"] = finish_reason;
        }
        rootNode["choices"] = choicesNode;
        
        Json::Value usageNode(Json::objectValue);
        usageNode["prompt_tokens"] = (Json::Int)input_token_count;
        usageNode["completion_tokens"] = (Json::Int)completion_tokens;
        usageNode["total_tokens"] = total_tokens;
        
        Json::Value promptTokenDetailsNode(Json::objectValue);
        promptTokenDetailsNode["cached_tokens"] = 0;
        promptTokenDetailsNode["audio_tokens"] = 0;
        usageNode["prompt_tokens_details"] = promptTokenDetailsNode;
        
        Json::Value completionTokenDetailsNode(Json::objectValue);
        completionTokenDetailsNode["reasoning_tokens"] = 0;
        completionTokenDetailsNode["audio_tokens"] = 0;
        completionTokenDetailsNode["accepted_prediction_tokens"] = 0;
        completionTokenDetailsNode["rejected_prediction_tokens"] = 0;
        usageNode["completion_tokens_details"] = completionTokenDetailsNode;
        
        rootNode["usage"] = usageNode;
        
    } catch (const std::exception& e) {
        throw;
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

/*
 The chat completion chunk object
 https://platform.openai.com/docs/api-reference/chat-streaming/streaming
 */
static std::string create_stream_chunk(int n,
                                       const std::string& id,
                                       const std::string& model,
                                       const std::string& content,
                                       const std::string& fingerprint,
                                       bool finish) {
    Json::Value root;
    root["id"] = id;
    root["object"] = "chat.completion.chunk";
    root["created"] = (Json::UInt64)std::time(nullptr);
    root["model"] = model;
    root["system_fingerprint"] = fingerprint;//Deprecated
    
    Json::Value choice;
    choice["index"] = n;
    
    Json::Value delta;
    if (content.empty() && !finish) {
        delta["role"] = "assistant";
    } else {
        delta["content"] = content;
    }
    delta["logprobs"] = Json::nullValue;
    choice["delta"] = delta;
    
    if (finish) {
        choice["finish_reason"] = "stop";
    } else {
        choice["finish_reason"] = Json::nullValue;
    }
    root["choices"].append(choice);
    
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return "data: " + Json::writeString(writer, root) + "\n\n";
}

static void run_inference_stream(
                                 OgaModel* model,
                                 OgaTokenizer* tokenizer,
                                 const std::string& modelName,
                                 const std::string& fingerprint,
                                 long long created,
                                 unsigned int max_tokens,
                                 unsigned int top_k,
                                 double top_p,
                                 double temperature,
                                 unsigned int n,
                                 std::string prompt,
                                 std::function<bool(const std::string&, unsigned int)> on_token_generated
                                 ) {
    
    // Create Tokenizer Stream
    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
    
    size_t input_token_count = 0;
    double max_length = 0;
    
    // Encode Prompt
    auto input_sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *input_sequences);
    input_token_count = input_sequences->SequenceCount(0);
    max_length = (double)(input_token_count + max_tokens);
    
    // Set Generation Parameters
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", max_length);
    params->SetSearchOption("top_k", top_k);
    params->SetSearchOption("top_p", top_p);
    params->SetSearchOption("temperature", temperature);
    params->SetSearchOption("num_return_sequences", n);
    
    // Create Generator
    // Generator is stateful; we need 1 per request.
    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*input_sequences);
    
    // Create a vector of streams
    // Decoding is stateful; we need 1 decoder per sequence.
    std::vector<std::string> generated_responses(n);
    std::vector<std::unique_ptr<OgaTokenizerStream>> streams;
    for (int i = 0; i < n; i++) {
        streams.push_back(OgaTokenizerStream::Create(*tokenizer));
    }
    
    // Start Generating
    while (1) {
        generator->GenerateNextToken();
        if(generator->IsDone()) break;
        // Iterate through each sequence (0 to n-1) to collect results
        for (int i = 0; i < n; i++) {
            // Get the full sequence data for the i-th choice
            const auto* seq_data = generator->GetSequenceData(i);
            size_t seq_len = generator->GetSequenceCount(i);
            // Safety check to ensure we have data
            if (seq_len == 0) continue;
            // Get the most recently generated token
            int32_t new_token = seq_data[seq_len - 1];
            // Decode using the specific stream for this sequence
            const char* token_str = streams[i]->Decode(new_token);
            if (token_str) {
                if (!on_token_generated(token_str, i)) {
                    // If callback returns false, client disconnected
                    break;
                }
            }
        }
    }
}

#pragma mark -

int main(int argc, OPTARG_T argv[]) {
    
    std::string model_path;           // -m
    std::string embedding_model_path; // -e
    OPTARG_T input_path  = NULL;      // -i
    OPTARG_T output_path = NULL;      // -o
    
    // Server mode flags
    bool server_mode = false;         // -s
    int port = 8080;                  // -p
    std::string host = "127.0.0.1";   // -h
    
    std::vector<unsigned char> cli_request_json(0);
    
    int ch;
    
    while ((ch = getopt(argc, argv, ARGS)) != -1) {
        switch (ch){
            case 'm':
#if WIN32
                model_path = wchar_to_utf8(optarg);
#else
                model_path = optarg;
#endif
                break;
            case 'e':
#if WIN32
                embedding_model_path = wchar_to_utf8(optarg);
#else
                embedding_model_path = optarg;
#endif
                break;
            case 'i':
                input_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 's':
                server_mode = true;
                break;
            case 'p':
                port = std::stoi(optarg);
                break;
            case 'h':
#if WIN32
                host = wchar_to_utf8(optarg);
#else
                host = optarg;
#endif
                break;
            case '-':
            {
                // Only relevant for CLI mode
                std::vector<uint8_t> buf(BUFLEN);
                size_t s;
                while ((s = fread(buf.data(), 1, buf.size(), stdin)) > 0) {
                    cli_request_json.insert(cli_request_json.end(), buf.begin(), buf.begin() + s);
                }
            }
                break;
            default:
                usage();
                break;
        }
    }
    
    std::string fingerprint;
    long long model_created = 0;
    std::string modelName;
    std::unique_ptr<OgaModel> model;
    std::unique_ptr<OgaTokenizer> tokenizer;
    
    std::string embedding_fingerprint;
    long long embedding_model_created = 0;
    std::string embedding_modelName;
    std::unique_ptr<Ort::Session> embeddings_session;
    size_t max_seq_len = 384;
    
    if (model_path.length() != 0) {
        // 1.a Initialize Model and Tokenizer (Load once)
        std::cerr << "[Chat] Loading from " << model_path << std::endl;
        fingerprint = get_system_fingerprint(model_path, "directml");
        modelName = get_model_name(model_path);
        try {
            model = OgaModel::Create(model_path.c_str());
            tokenizer = OgaTokenizer::Create(*model);
            model_created = get_created_timestamp();
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            return 1;
        }
    }
    
    if (embedding_model_path.length() != 0) {
        // 1.b Initialize Embedding and Session (Load once)
        std::cerr << "[Embedding] Loading from " << embedding_model_path << std::endl;
        embedding_fingerprint = get_system_fingerprint(embedding_model_path, "directml");
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Embedding");
        
        try {
            embedding_modelName = get_model_name(embedding_model_path);
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            //            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*)session_options, OrtGetApiBase()));
            embeddings_session = std::make_unique<Ort::Session>(env, embedding_model_path.c_str(), session_options);
            GetMaxSeqLen(embeddings_session.get(), &max_seq_len);
            embedding_model_created = get_created_timestamp();
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            return 1;
        }
    }
    
    size_t num_input_nodes = embeddings_session->GetInputCount();
    size_t num_output_nodes = embeddings_session->GetOutputCount();
    
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_shape = {1}; // Batch size 1
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name_ptr = embeddings_session->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name_ptr.get());
        std::cout << "Input " << i << " Name: " << input_name_ptr.get() << std::endl;
    }
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name_ptr = embeddings_session->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name_ptr.get());
        std::cout << "Input " << i << " Name: " << output_name_ptr.get() << std::endl;
    }
    
    // 1. Convert std::string names to const char* array
    std::vector<const char*> input_names_c_array;
    for (const auto& name : input_node_names) {
        input_names_c_array.push_back(name.c_str());
    }
    
    std::vector<const char*> output_names_c_array;
    for (const auto& name : output_node_names) {
        output_names_c_array.push_back(name.c_str());
    }
    
    try {
        if(true) {
            
            const char* input_text_arr[] = { "This is the text to embed." };
            
            // --- STEP 1: PREPARE DATA ---
            const char* text_val = "Hello world";
            const char* input_strings[] = { text_val };
            size_t batch_size = 1;
            int64_t input_shape[] = { (int64_t)batch_size };
            
            // --- STEP 2: GET C-API ACCESS ---
            const OrtApi& api = Ort::GetApi();
            
            // --- STEP 3: CREATE EMPTY TENSOR ---
            OrtAllocator* allocator;
            api.GetAllocatorWithDefaultOptions(&allocator);
            OrtValue* raw_tensor_ptr = nullptr;
            OrtStatus* status = api.CreateTensorAsOrtValue(
                                                           allocator,                            // 1. Allocator
                                                           input_shape,                          // 2. Shape
                                                           1,                                    // 3. Shape Rank
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, // 4. Type
                                                           &raw_tensor_ptr                       // 5. Output
                                                           );
            if (status != nullptr) {
                std::cerr << "Creation failed: " << api.GetErrorMessage(status) << std::endl;
                api.ReleaseStatus(status);
            }else{
                // --- STEP 4: FILL TENSOR WITH STRINGS ---
                // Now we copy the C-strings into the allocated tensor
                status = api.FillStringTensor(
                                              raw_tensor_ptr,                       // Tensor to fill
                                              input_strings,                        // Array of C-strings
                                              batch_size                            // Number of strings
                                              );
                if (status != nullptr) {
                    std::cerr << "Filling failed: " << api.GetErrorMessage(status) << std::endl;
                    api.ReleaseStatus(status);
                }else{
                    // --- STEP 5: CONVERT TO C++ OBJECT ---
                    // Ort::Value takes ownership and handles cleanup
                    Ort::Value input_tensor(raw_tensor_ptr);
                    
                    auto outputs = embeddings_session->Run(
                                                           Ort::RunOptions{nullptr},
                                                           input_names_c_array.data(),
                                                           &input_tensor,
                                                           num_input_nodes,
                                                           output_names_c_array.data(),
                                                           num_output_nodes
                                                           );
                    std::cout << "Inference Successful!" << std::endl;
                    
                    //                    auto _outputs = embeddings_session->GetOutputs();
                    
                    size_t dimensions = outputs.size();
                    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
                    auto shape = output_info.GetShape();
                    
                    // 2. Identify the Embedding Dimension
                    // If shape is [512], then dim is 512.
                    int64_t embedding_dim = shape[1];
                    
                    float* floatarr = outputs[0].GetTensorMutableData<float>();
                    // 3. Create the std::vector
                    std::vector<float> embeddings(floatarr, floatarr + embedding_dim);
                    
                    /*
                     
                     https://huggingface.co/SamLowe/universal-sentence-encoder-large-5-onnx
                     
                     It uses the ONNXRuntime Extensions to embed the tokenizer within the ONNX model, so no seperate tokenizer is needed, and text is fed directly into the ONNX model.
                     
                     Post-processing (E.g. pooling, normalization) is also implemented within the ONNX model, so no separate processing is necessary.
                     */
                    
                    std::cout << "Final Embedding: [ ";
                    for(int i = 0; i < embedding_dim; ++i) std::cout << embeddings[i] << " ";
                    std::cout << "... ]" << std::endl;
                    
                    
                    if(false) {
                        float sum_squares = 0.0f;
                        for (float val : embeddings) {
                            sum_squares += val * val;
                        }
                        // Calculate Norm
                        float norm = std::sqrt(sum_squares);
                        // Normalize the vector
                        // (Check for near-zero to avoid NaN, though unlikely with embeddings)
                        if (norm > 1e-9) {
                            for (size_t i = 0; i < embedding_dim; ++i) {
                                embeddings[i] /= norm;
                            }
                        }
                        
                    }
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    //                    for (int i = 0; i < seq_len; i++) {
                    //                        for (int j = 0; j < hidden_size; j++) {
                    //                            // Safe access using dynamic seq_len
                    //                            embedding[j] += float_data[i * hidden_size + j];
                    //                        }
                    //                    }
                    
                    // Average
                    //                    if (seq_len > 0) {
                    //                        for (float& val : embedding) val /= static_cast<float>(seq_len);
                    //                    }
                    
                    // 3. L2 Normalization (Optional, but recommended for cosine similarity)
                    //                    float sum_squares = 0.0f;
                    //                    for (float val : embedding) sum_squares += val * val;
                    //                    float magnitude = std::sqrt(sum_squares);
                    //
                    //                    if (magnitude > 1e-9) {
                    //                        for (float& val : embedding) val /= magnitude;
                    //                    }
                    
                    // 4. Print Result
                    //                    std::cout << "Final Embedding: [ ";
                    //                    for(int i=0; i<5; ++i) std::cout << embedding[i] << " ";
                    //                    std::cout << "... ]" << std::endl;
                    //
                    
                    
                    
                    
                    
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    
    // ---------------------------------------------------------
    // SERVER MODE
    // ---------------------------------------------------------
    if (server_mode) {
        httplib::Server svr;
        
        // Route: /v1/chat/completions
        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/chat/completions request received." << std::endl;
            
            try {
                
                std::string request_body;
                std::string prompt;
                unsigned int max_tokens = 2048;
                unsigned int top_k = 50;
                double top_p = 0.9;
                double temperature = 0.7;
                unsigned int n = 1;
                bool is_stream = false;
                
                before_run_inference(req.body,
                                     prompt,
                                     &max_tokens,
                                     &top_k,
                                     &top_p,
                                     &temperature,
                                     &n,
                                     &is_stream);
                
                if(is_stream) {
                    std::string req_id = generate_openai_style_id();
                    
                    // Corrected Lambda structure
                    res.set_chunked_content_provider("text/event-stream",
                                                     [&, req_id, prompt, max_tokens, top_k, top_p, temperature, n ](size_t offset, httplib::DataSink &sink) {
                        // Send initial role packet (optional but good practice)
                        for (int i = 0; i < n; i++) {
                            std::string role_chunk = create_stream_chunk(i, req_id, modelName, fingerprint, "");
                            sink.write(role_chunk.data(), role_chunk.size());
                        }
                        
                        // Define a callback to handle tokens as they are generated
                        auto token_callback = [&](const std::string& token, unsigned int n) {
                            std::string chunk = create_stream_chunk(n, req_id, modelName, fingerprint, token);
                            sink.write(chunk.data(), chunk.size());
                            return true; // Return false to stop inference if needed
                        };
                        
                        // Run Inference (You must implement run_inference_stream)
                        // Note: This function must block here until finished, calling token_callback repeatedly
                        run_inference_stream(
                                             model.get(),
                                             tokenizer.get(),
                                             modelName,
                                             fingerprint,
                                             model_created,
                                             max_tokens,
                                             top_k,
                                             top_p,
                                             temperature,
                                             n,
                                             prompt,
                                             token_callback
                                             );
                        // 4. Send finish reason
                        std::string finish_chunk = create_stream_chunk(n, req_id, modelName, fingerprint, "", true);
                        sink.write(finish_chunk.data(), finish_chunk.size());
                        
                        // 5. Send [DONE] to close the stream for the client
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.data(), done.size());
                        
                        sink.done(); // Close the connection
                        return true;
                    }
                                                     );
                    
                }else{
                    // Run Inference
                    std::string response_json = run_inference(
                                                              model.get(),
                                                              tokenizer.get(),
                                                              modelName,
                                                              fingerprint,
                                                              model_created,
                                                              max_tokens,
                                                              top_k,
                                                              top_p,
                                                              temperature,
                                                              n,
                                                              prompt
                                                              );
                    res.set_content(response_json, "application/json");
                    res.status = 200;
                }
            } catch (const std::exception& e) {
                // Build Error JSON
                Json::Value rootNode(Json::objectValue);
                Json::Value errorNode(Json::objectValue);
                errorNode["message"] = e.what();
                errorNode["type"] = "invalid_request_error";
                errorNode["param"] = Json::nullValue;
                errorNode["code"] = Json::nullValue;
                rootNode["error"] = errorNode;
                
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "";
                std::string error_str = Json::writeString(writer, rootNode);
                
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
        });
        
        // Route: /v1/models
        svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
            std::cout << "[Server] /v1/models request received." << std::endl;
            
            // 2. Create the list wrapper
            Json::Value root(Json::objectValue);
            root["object"] = "list";
            root["data"] = Json::Value(Json::arrayValue);
            
            // 1. Create the model object
            if(model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = modelName;
                modelCard["object"] = "model";
                modelCard["created"] = model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            if(embedding_model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = embedding_modelName;
                modelCard["object"] = "model";
                modelCard["created"] = embedding_model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            
            // 3. Serialize
            Json::StreamWriterBuilder writer;
            writer["indentation"] = ""; // Minified JSON
            std::string json_str = Json::writeString(writer, root);
            
            // 4. Respond
            res.set_content(json_str, "application/json");
            res.status = 200;
        });
        
        // Route: /v1/embeddings
        svr.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/embeddings request received." << std::endl;
            
            try {
                
                
                
                
                //                res.set_content(response_json, "application/json");
                res.status = 200;
                
            } catch (const std::exception& e) {
                // Build Error JSON
                Json::Value rootNode(Json::objectValue);
                Json::Value errorNode(Json::objectValue);
                errorNode["message"] = e.what();
                errorNode["type"] = "invalid_request_error";
                errorNode["param"] = Json::nullValue;
                errorNode["code"] = Json::nullValue;
                rootNode["error"] = errorNode;
                
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "";
                std::string error_str = Json::writeString(writer, rootNode);
                
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
            
        });
        
        std::cout << "[Server] Listening on " << host << ":" << port << std::endl;
        
        // Listen (Blocking call)
        if (!svr.listen(host.c_str(), port)) {
            std::cerr << "Error: Could not start server on " << host << ":" << port << std::endl;
            return 1;
        }
    }
    // ---------------------------------------------------------
    // CLI MODE
    // ---------------------------------------------------------
    else {
        // Handle input file reading if not piped via stdin ('-')
        if ((!cli_request_json.size()) && (input_path != NULL)) {
            FILE *f = _fopen(input_path, _rb);
            if(f) {
                fseek(f, 0, SEEK_END);
                size_t len = (size_t)ftell(f);
                fseek(f, 0, SEEK_SET);
                cli_request_json.resize(len);
                fread(cli_request_json.data(), 1, cli_request_json.size(), f);
                fclose(f);
            }
        }
        
        if (cli_request_json.size() == 0) {
            usage();
            return 1;
        }
        
        std::string request_str((const char *)cli_request_json.data(), cli_request_json.size());
        std::string response;
        
        try {
            
            std::string request_body;
            std::string prompt;
            unsigned int max_tokens = 2048;
            unsigned int top_k = 50;
            double top_p = 0.9;
            double temperature = 0.7;
            unsigned int n = 1;
            bool is_stream = false;
            
            before_run_inference(request_body,
                                 prompt,
                                 &max_tokens,
                                 &top_k,
                                 &top_p,
                                 &temperature,
                                 &n,
                                 &is_stream);
            
            response = run_inference(
                                     model.get(),
                                     tokenizer.get(),
                                     modelName,
                                     fingerprint,
                                     model_created,
                                     max_tokens,
                                     top_k,
                                     top_p,
                                     temperature,
                                     n,
                                     prompt
                                     );
            
        } catch (const std::exception& e) {
            // CLI Error Format
            Json::Value rootNode(Json::objectValue);
            Json::Value errorNode(Json::objectValue);
            rootNode["error"] = errorNode;
            errorNode["message"] = e.what();
            errorNode["type"] = "invalid_request_error";
            
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            response = Json::writeString(writer, rootNode);
        }
        
        // Output logic
        if(!output_path) {
            std::cout << response << std::endl;
        } else {
            FILE *f = _fopen(output_path, _wb);
            if(f) {
                fwrite(response.c_str(), 1, response.length(), f);
                fclose(f);
            }
        }
    }
    
    return 0;
}
