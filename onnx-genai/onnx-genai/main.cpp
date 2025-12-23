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

static void usage(void)
{
    fprintf(stderr, "Usage:  onnx-genai -m model -i input\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c path     : %s\n", 'm' , "model");
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
#define ARGS (OPTARG_T)L"m:i:o:sp:-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:i:o:sp:-h"
#define _atoi atoi
#define _atof atof
#endif

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
              unsigned int *n) {
    
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
        }
    }
}

/*
 * Encapsulates the core inference logic to be used by both CLI and Server modes.
 */
std::string run_inference(
    OgaModel* model,
    OgaTokenizer* tokenizer,
    const std::string& modelName,
    const std::string& fingerprint,
    const std::string& request_body
) {
    // Default parameters
    unsigned int max_tokens = 2048;
    unsigned int top_k = 50;
    double top_p = 0.9;
    double temperature = 0.7;
    unsigned int n = 1;
    
    std::string prompt;
    
    // Parse the input JSON
    // Note: If parse_request fails/throws, it should be caught by the caller
    parse_request(request_body, prompt, &max_tokens, &top_k, &top_p, &temperature, &n);

    std::string content;
    Json::Value rootNode(Json::objectValue);
    size_t completion_tokens = 0;
    size_t input_token_count = 0;
    std::string finish_reason = "stop";
    double max_length = 0;

    try {
        // Create Tokenizer Stream
        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

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
                
        // Create Generator (Generator is stateful and must be created per request)
        auto generator = OgaGenerator::Create(*model, *params);
        generator->AppendTokenSequences(*input_sequences);

        // Start Generating
        while (1) {
            generator->GenerateNextToken();
            
            if(generator->IsDone()) break;
            
            auto seq_data = generator->GetSequenceData(0);
            size_t seq_len = generator->GetSequenceCount(0);
            int32_t new_token = seq_data[seq_len - 1];
            const char* token_str = tokenizer_stream->Decode(new_token);
            if (token_str) {
                content += token_str;
            }
            completion_tokens++;
        }
        
        Json::Int total_tokens = (Json::Int)(input_token_count + completion_tokens);
        if (total_tokens >= max_length) {
            finish_reason = "length";
        }
        
        // Build Response JSON
        rootNode["id"] = generate_openai_style_id();
        rootNode["object"] = "chat.completion";
        rootNode["created"] = get_created_timestamp();
        rootNode["model"] = modelName;
        rootNode["system_fingerprint"] = fingerprint;
        
        Json::Value choicesNode(Json::arrayValue);
        Json::Value choiceNode(Json::objectValue);
        choiceNode["index"] = 0;
        choiceNode["logprobs"] = Json::nullValue;
        choiceNode["finish_reason"] = finish_reason;
        
        Json::Value messageNode(Json::objectValue);
        messageNode["role"] = "assistant";
        messageNode["content"] = content;
        messageNode["refusal"] = Json::nullValue;
        choiceNode["message"] = messageNode;
        
        choicesNode.append(choiceNode);
        rootNode["choices"] = choicesNode;
        
        Json::Value usageNode(Json::objectValue);
        usageNode["prompt_tokens"] = (Json::Int)input_token_count;
        usageNode["completion_tokens"] = (Json::Int)completion_tokens;
        usageNode["total_tokens"] = total_tokens;
        
        // Details
        Json::Value usageDetailsNode(Json::objectValue);
        usageDetailsNode["reasoning_tokens"] = 0;
        usageDetailsNode["accepted_prediction_tokens"] = 0;
        usageDetailsNode["rejected_prediction_tokens"] = 0;
        usageNode["completion_tokens_details"] = usageDetailsNode;
        
        rootNode["usage"] = usageNode;

    } catch (const std::exception& e) {
        // Rethrow to be handled by the caller (CLI or Server)
        throw;
    }

    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

// ------------------------------------------------

int main(int argc, OPTARG_T argv[]) {
        
    std::string model_path;           // -m
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
    
    if (model_path.length() == 0) {
        usage();
        return 1;
    }

    // 1. Initialize Model and Tokenizer (Load once)
    std::cerr << "[Model] Loading from " << model_path << std::endl;
    std::string fingerprint = get_system_fingerprint(model_path, "directml");
    std::string modelName = get_model_name(model_path);
    
    std::unique_ptr<OgaModel> model;
    std::unique_ptr<OgaTokenizer> tokenizer;

    try {
        model = OgaModel::Create(model_path.c_str());
        tokenizer = OgaTokenizer::Create(*model);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return 1;
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
                
                // Run Inference
                std::string response_json = run_inference(
                    model.get(),
                    tokenizer.get(),
                    modelName,
                    fingerprint,
                    req.body
                );

                res.set_content(response_json, "application/json");
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

        // Route: /v1/models
        svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
            std::cout << "[Server] /v1/models request received." << std::endl;
            
            // 1. Create the model object
            Json::Value modelCard(Json::objectValue);
            modelCard["id"] = modelName;
            modelCard["object"] = "model";
            modelCard["created"] = (Json::UInt64)std::time(nullptr); 
            modelCard["owned_by"] = "system";
            
            // 2. Create the list wrapper
            Json::Value root(Json::objectValue);
            root["object"] = "list";
            root["data"] = Json::Value(Json::arrayValue);
            root["data"].append(modelCard);
            
            // 3. Serialize
            Json::StreamWriterBuilder writer;
            writer["indentation"] = ""; // Minified JSON
            std::string json_str = Json::writeString(writer, root);
            
            // 4. Respond
            res.set_content(json_str, "application/json");
            res.status = 200;
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
            response = run_inference(
                model.get(),
                tokenizer.get(),
                modelName,
                fingerprint,
                request_str
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
