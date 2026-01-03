//
//  main.cpp
//  onnx-genai
//
//  Created by miyako on 2025/09/03.
//

#include "onnx-genai.h"

namespace fs = std::filesystem;
using namespace tokenizers; // mlc-ai namespace

static // Helper: Read entire file into a string (Blob)
std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (!fs) throw std::runtime_error("Could not open file: " + path);
    
    std::string data((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    return data;
}

static // Unified Loader
std::unique_ptr<Tokenizer> LoadTokenizer(const std::string& model_path) {
    fs::path path(model_path);
    
    // 1. Check if the path points to a directory or a specific file
    fs::path json_path = path;
    fs::path model_file_path = path;

    if (fs::is_directory(path)) {
        // If user gave a folder, look for standard names
        json_path = path / "tokenizer.json";
        model_file_path = path / "tokenizer.model";
    }

    // 2. Try to load Hugging Face JSON first (preferred for modern models)
    if (fs::exists(json_path) && json_path.extension() == ".json") {
        std::cout << "Loading HF Tokenizer from: " << json_path << std::endl;
        std::string blob = LoadBytesFromFile(json_path.string());
        return Tokenizer::FromBlobJSON(blob);
    }
    
    // 3. Fallback to SentencePiece
    if (fs::exists(model_file_path) && model_file_path.extension() == ".model") {
        std::cout << "Loading SentencePiece from: " << model_file_path << std::endl;
        std::string blob = LoadBytesFromFile(model_file_path.string());
        return Tokenizer::FromBlobSentencePiece(blob);
    }

    return 0;
}

#ifdef WIN32
static std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();

    // Get required buffer size in characters (including null terminator)
    int size_needed = MultiByteToWideChar(
        CP_UTF8,       // Source is UTF-8
        0,             // Default flags
        str.c_str(),   // Source string
        -1,            // Null-terminated
        nullptr,       // No output buffer yet
        0              // Requesting size
    );

    if (size_needed <= 0) return std::wstring();

    // Allocate buffer
    std::wstring wstr(size_needed, 0);

    // Perform conversion
    MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        -1,
        &wstr[0],
        size_needed
    );

    // Remove the extra null terminator added by MultiByteToWideChar
    if (!wstr.empty() && wstr.back() == '\0') {
        wstr.pop_back();
    }

    return wstr;
}

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

// Improved signature: uses Eigen::Ref to avoid copies if passing blocks/maps
Eigen::VectorXf mean_pool(
    const Eigen::Ref<const Eigen::MatrixXf>& hidden,
    const Eigen::Ref<const Eigen::VectorXi>& mask
) {
    // 1. Safety Check
    if (hidden.rows() != mask.size()) {
        throw std::invalid_argument("Hidden state sequence length does not match mask length.");
    }

    // 2. Convert mask to float for matrix multiplication
    // Casting is usually very fast compared to the accumulation logic
    Eigen::VectorXf mask_f = mask.cast<float>();

    // 3. Calculate Count (Sum of mask)
    float count = mask_f.sum();
    
    // Edge case: empty mask
    if (count <= 0.0f) {
        return Eigen::VectorXf::Zero(hidden.cols());
    }

    // 4. Matrix Multiplication approach (The main optimization)
    // Formula: (1/N) * (mask^T * Hidden)
    //
    // mask_f             is [seq_len, 1]
    // hidden             is [seq_len, hidden_dim]
    // mask_f.transpose() is [1, seq_len]
    // result             is [1, hidden_dim]
    
    // Note: We create a temporary row vector, then transpose it back
    // to match the return type (VectorXf is a column vector).
    Eigen::VectorXf pooled = (mask_f.transpose() * hidden).transpose();

    return pooled / count;
}

Eigen::MatrixXf mean_pool_batch(
    const std::vector<Eigen::MatrixXf>& hidden_batch,
    const std::vector<Eigen::VectorXi>& mask_batch
) {
    // 1. Safety Checks
    if (hidden_batch.empty()) {
        return Eigen::MatrixXf(0, 0);
    }
    if (hidden_batch.size() != mask_batch.size()) {
        throw std::invalid_argument("Batch size mismatch between hidden states and masks.");
    }

    long batch_size = hidden_batch.size();
    long hidden_dim = hidden_batch[0].cols();

    // Allocate the result matrix once
    Eigen::MatrixXf out(batch_size, hidden_dim);

    // 2. Parallel Processing (OpenMP)
    // This distributes the rows across available CPU cores.
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        
        // --- Step A: Optimized Mean Pooling (Inlined) ---
        // We write directly into out.row(i) to avoid creating temporary VectorXf objects.
        
        const auto& hidden = hidden_batch[i];
        const auto& mask = mask_batch[i];

        // Convert mask to float for calculation
        Eigen::VectorXf mask_f = mask.cast<float>();
        float count = mask_f.sum();

        if (count > 0.0f) {
            // Matrix Mult: [1, seq] * [seq, dim] -> [1, dim]
            // We assign this directly to the output row.
            out.row(i) = mask_f.transpose() * hidden;
            out.row(i) /= count;
            
            // --- Step B: Optimized L2 Normalize (In-Place) ---
            // Calculate norm of the row we just wrote
            float norm = out.row(i).norm();
            
            if (norm > 1e-12f) {
                out.row(i) /= norm;
            }
        } else {
            // Handle edge case: empty mask -> zero vector
            out.row(i).setZero();
        }
    }

    return out;
}

Eigen::VectorXf l2_normalize(const Eigen::Ref<const Eigen::VectorXf>& v) {
    float norm = v.norm();
    // Use a small epsilon to prevent division by near-zero values
    // and ensure numerical stability.
    if (norm > 1e-12f)
        return v.normalized(); // Uses Eigen's optimized internal implementation
    // If norm is effectively zero, return the original (zero) vector
    return v;
}

#pragma mark -

static void usage(void)
{
    fprintf(stderr, "Usage:  onnx-genai -m model -i input\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c path     : %s\n", 'm' , "model");
    fprintf(stderr, " -%c path     : %s\n", 'e' , "embedding model");
    fprintf(stderr, " -%c template : %s\n", 'c' , "chat template");
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
#define ARGS (OPTARG_T)L"m:e:i:o:sp:c:-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:e:i:o:sp:c:-h"
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

static std::string get_openai_style_id() {
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

#pragma mark -

static void parse_request_embeddings(const std::string &json,
                                     std::string &input) {
    
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
            Json::Value input_node = root["input"];
            if(input_node.isString())
            {
                input = input_node.asString();
            }
        }
    }
}

static void parse_request(
                          const std::string &json,
                          std::string &prompt,
                          unsigned int *max_tokens,
                          unsigned int *top_k,
                          double *top_p,
                          double *temperature,
                          unsigned int *n,
                          bool *is_stream,
                          OgaTokenizer* tokenizer,
                          std::string& chat_template) {
    
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
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "";
                std::string messages_json = Json::writeString(writer, messages_node);

                prompt = tokenizer->ApplyChatTemplate(chat_template.c_str(), messages_json.c_str(), nullptr, true);
                
                std::cout << prompt << std::endl;
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

static void before_run_embeddings(
                                  const std::string& request_body,
                                  std::string &input
                                  ) {
    parse_request_embeddings(request_body, input);
}

static void before_run_inference(
                                 const std::string& request_body,
                                 std::string &prompt,
                                 unsigned int *max_tokens,
                                 unsigned int *top_k,
                                 double *top_p,
                                 double *temperature,
                                 unsigned int *n,
                                 bool *is_stream,
                                 OgaTokenizer* tokenizer,
                                 std::string& chat_template) {
    
    parse_request(request_body, prompt, max_tokens, top_k, top_p, temperature, n, is_stream, tokenizer, chat_template);
}

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
        rootNode["id"] = get_openai_style_id();
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
                // Skip chat end tokens
                if (
                    strcmp(token_str, "<|im_end|>") == 0 ||
                    strcmp(token_str, "</s>") == 0 ||
                    strcmp(token_str, "<|eot_id|>") == 0 ||
                    strcmp(token_str, "<</SYS>>") == 0 ||
                    strcmp(token_str, "[/INST]") == 0 ||
                    strcmp(token_str, "END_OF_TURN_TOKEN") == 0 ||
                    strcmp(token_str, "<end_of_turn>") == 0
                    ) {
                        continue;
                    }
                if (!on_token_generated(token_str, i)) {
                    // If callback returns false, client disconnected
                    break;
                }
            }
        }
    }
}

static // Helper to convert int32 -> int64
std::vector<int64_t> ConvertToInt64(const std::vector<int>& input_ids) {
    std::vector<int64_t> output(input_ids.size());
    std::transform(input_ids.begin(), input_ids.end(), output.begin(),
                   [](int i) { return static_cast<int64_t>(i); });
    return output;
}

static // --- The Processor Code ---
void ProcessOutput(
    Ort::Value& output_tensor,              // From session.Run()
    const std::vector<int64_t>& input_mask, // The flat mask you sent to ONNX
    int64_t batch_size,
    int64_t seq_len
) {
    // 1. Get Output Data Info
    // Shape is typically [BatchSize, SeqLen, HiddenSize]
    auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    int64_t hidden_size = shape[2];
    
    // Get pointer to the raw float array containing all batches
    float* raw_data = output_tensor.GetTensorMutableData<float>();

    // 2. Prepare Containers for your function
    std::vector<Eigen::MatrixXf> hidden_batch_vec;
    std::vector<Eigen::VectorXi> mask_batch_vec;

    hidden_batch_vec.reserve(batch_size);
    mask_batch_vec.reserve(batch_size);

    // 3. Loop over Batch to slice data
    for (int i = 0; i < batch_size; ++i) {
        // --- A. Extract Hidden States ---
        // Calculate offset for this specific batch item
        float* batch_ptr = raw_data + (i * seq_len * hidden_size);

        // Map raw memory to Eigen.
        // IMPORTANT: ONNX is RowMajor, Eigen Default is ColMajor.
        // We Map as RowMajor, then assign to MatrixXf (which copies/converts to ColMajor).
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mapped_matrix(batch_ptr, seq_len, hidden_size);
        
        hidden_batch_vec.push_back(mapped_matrix);

        // --- B. Extract/Convert Mask ---
        // Input mask was flat [Batch * Seq]. We slice it for this batch.
        std::vector<int> current_mask_int;
        current_mask_int.reserve(seq_len);
        
        for(int j = 0; j < seq_len; ++j) {
            // Convert int64_t -> int (Eigen::VectorXi is int)
            current_mask_int.push_back(static_cast<int>(input_mask[i * seq_len + j]));
        }
        
        // Map std::vector to Eigen::VectorXi
        mask_batch_vec.push_back(
            Eigen::Map<Eigen::VectorXi>(current_mask_int.data(), seq_len)
        );
    }

    // 4. Perform Mean Pooling
    // Result is likely [BatchSize, HiddenSize] (depending on your implementation)
    Eigen::MatrixXf pooled_embeddings = mean_pool_batch(hidden_batch_vec, mask_batch_vec);

    // 5. Perform L2 Normalization
    // We iterate through the rows (or cols) of the pooled result
    // Assuming pooled_embeddings is [BatchSize, HiddenSize] (Row per embedding)
    for (int i = 0; i < pooled_embeddings.rows(); ++i) {
        // Extract the row as a vector
        Eigen::VectorXf emb = pooled_embeddings.row(i);
        
        // Normalize
        Eigen::VectorXf normalized_emb = l2_normalize(emb);
        
        // (Optional) Store it back or print it
        std::cout << "Normalized Embedding " << i << ":\n"
                  << normalized_emb.head(5) // Print first 5 dims
                  << "...\n" << std::endl;
    }
}

static std::string run_embeddings(
                                  Ort::Session *session,
                                  std::vector<int>& ids,
                                  std::vector<const char*>&  input_names_c_array,
                                  size_t num_input_nodes,
                                  std::vector<const char*>&   output_names_c_array,
                                      size_t num_output_nodes) {

    int batch_size = 1;
    std::vector<int64_t> input_ids = ConvertToInt64(ids);
    int seq_len = (int)ids.size();
    
    Json::Value rootNode(Json::objectValue);
    
    try {
        
        std::vector<int64_t> input_node_dims = {batch_size, (int64_t)input_ids.size()};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // Define Shapes
        std::vector<int64_t> input_dims = {batch_size, seq_len};
        // Create Attention Mask (1 for real tokens)
        std::vector<int64_t> attention_mask(seq_len, 1);
        // Create the Missing Vector (All Zeros)
        std::vector<int64_t> token_type_ids(seq_len, 0);
        // Create Inputs Vector
        std::vector<Ort::Value> input_tensors;
        
        // Mistral / Llama / Qwen: only need input_ids
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                                                                  memory_info,
                                                                  input_ids.data(),
                                                                  input_ids.size(),
                                                                  input_dims.data(),
                                                                  input_dims.size()));
        if (num_input_nodes >1) {
            // DistilBERT: only needs input_ids and attention_mask.
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                                                                      memory_info,
                                                                      attention_mask.data(),
                                                                      attention_mask.size(),
                                                                      input_dims.data(),
                                                                      input_dims.size()));
        }
        if (num_input_nodes >2) {
            // BERT / RoBERTa / MiniLM: ALWAYS require token_type_ids
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                                                                      memory_info,
                                                                      token_type_ids.data(),
                                                                      token_type_ids.size(),
                                                                      input_dims.data(),
                                                                      input_dims.size()));
        }

        auto outputs = session->Run(
                                    Ort::RunOptions{nullptr},
                                    input_names_c_array.data(),
                                    input_tensors.data(),
                                    num_input_nodes,
                                    output_names_c_array.data(),
                                    num_output_nodes
                                    );
        
        size_t dimensions = outputs.size();
        if(dimensions > 0) {
            
            auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
            float* floatarr = outputs[0].GetTensorMutableData<float>();
            
            auto shape = output_info.GetShape();// [Batch, Seq, Hidden]
            if(shape.size() > 2) {
                int64_t hidden_size = shape[2];
                // 2. Prepare Batches
                std::vector<Eigen::MatrixXf> hidden_batch_vec;
                std::vector<Eigen::VectorXi> mask_batch_vec;
                // (Since batch=1, loop runs once)
                // Map raw data: ONNX (RowMajor) -> Eigen Map
                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                mapped_hidden(floatarr, seq_len, hidden_size);
                // Deep copy into MatrixXf (converts to ColMajor for Eigen internal efficiency)
                hidden_batch_vec.push_back(mapped_hidden);
                // Convert mask to Eigen::VectorXi
                Eigen::VectorXi mask_vec(seq_len);
                for(int i=0; i<seq_len; ++i) mask_vec(i) = (int)attention_mask[i];
                mask_batch_vec.push_back(mask_vec);
                // 1. Mean Pool
                Eigen::MatrixXf pooled = mean_pool_batch(hidden_batch_vec, mask_batch_vec); // Returns [1, Hidden]
                // 2. Normalize
                Eigen::VectorXf final_embedding = l2_normalize(pooled.row(0));
                // Create the std::vector
                std::vector<float> embeddings(final_embedding.data(), final_embedding.data() + final_embedding.size());
                rootNode["object"] = "embedding";
                Json::Value embeddingsNode(Json::arrayValue);
                for (float val : embeddings) {
                    embeddingsNode.append(val);
                }
                rootNode["embedding"] = embeddingsNode;
                rootNode["index"] = 0;
            }
        }
    } catch (const std::exception& e) {
        throw;
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

static std::string run_embeddings_e2e(
                                  Ort::Session *session,
                                  std::string& input,
                                  std::vector<const char*>&  input_names_c_array,
                                  size_t num_input_nodes,
                                  std::vector<const char*>&   output_names_c_array,
                                  size_t num_output_nodes) {

    Json::Value rootNode(Json::objectValue);
    try {
        const char* input_strings[] = { input.c_str() };
        size_t batch_size = 1;
        int64_t input_shape[] = { (int64_t)batch_size };
        const OrtApi& api = Ort::GetApi();
        OrtAllocator* allocator;
        OrtStatus* status = api.GetAllocatorWithDefaultOptions(&allocator);
        OrtValue* raw_tensor_ptr = nullptr;
        status = api.CreateTensorAsOrtValue(
                                                       allocator,                            // 1. Allocator
                                                       input_shape,                          // 2. Shape
                                                       1,                                    // 3. Shape Rank
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, // 4. Type
                                                       &raw_tensor_ptr                       // 5. Output
                                                       );
        if (status != nullptr) {
            std::cerr << "CreateTensorAsOrtValue() failed: " << api.GetErrorMessage(status) << std::endl;
            api.ReleaseStatus(status);
            return "";
        }
        status = api.FillStringTensor(
                                      raw_tensor_ptr,                       // Tensor to fill
                                      input_strings,                        // Array of C-strings
                                      batch_size                            // Number of strings
                                      );
        if (status != nullptr) {
            std::cerr << "FillStringTensor() failed: " << api.GetErrorMessage(status) << std::endl;
            api.ReleaseStatus(status);
            return "";
        }
        Ort::Value input_tensor(raw_tensor_ptr);
        auto outputs = session->Run(
                                    Ort::RunOptions{nullptr},
                                    input_names_c_array.data(),
                                    &input_tensor,
                                    num_input_nodes,
                                    output_names_c_array.data(),
                                    num_output_nodes
                                    );
        size_t dimensions = outputs.size();
        if(dimensions > 0) {
            
            auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
            float* floatarr  = outputs[0].GetTensorMutableData<float>();
            
            Json::Value embeddingsNode(Json::arrayValue);
            auto shape = output_info.GetShape();
            if(shape.size() > 0) {
                int64_t embedding_dim = shape[1];
                // Create the std::vector
                std::vector<float> embeddings(floatarr, floatarr + embedding_dim);
                rootNode["object"] = "embedding";
                for (float val : embeddings) {
                    embeddingsNode.append(val);
                }
            }
            rootNode["embedding"] = embeddingsNode;
            rootNode["index"] = 0;
        }
    } catch (const std::exception& e) {
        throw;
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

#pragma mark -

int main(int argc, OPTARG_T argv[]) {
    
#ifdef WIN32
    std::wstring model_path_u16;
    std::wstring embedding_model_path_u16;
#endif
    std::string model_path;           // -m
    std::string embedding_model_path; // -e
    std::string chat_template_name;   // -c
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
#ifdef WIN32
                model_path_u16 = optarg;
                model_path = wchar_to_utf8(model_path_u16.c_str());
#else
                model_path = optarg;
#endif
                break;
            case 'e':
#ifdef WIN32
                embedding_model_path_u16 = optarg;
                embedding_model_path = wchar_to_utf8(embedding_model_path_u16.c_str());
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
            case 'c':
                chat_template_name = optarg;
                break;
            case 's':
                server_mode = true;
                break;
            case 'p':
                port = std::stoi(optarg);
                break;
            case 'h':
#ifdef WIN32
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
    
    std::string chat_template;
    
    //default:
    chat_template = R"(
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{'<|im_start|>assistant\n'}}
{{'<think>'}}
{% endif %}
)";
    
    if(chat_template_name == "llama3") {
        chat_template = R"(
{% for message in messages %}
{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
{% endfor %}
{% if add_generation_prompt %}
{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}
{% endif %}
)";
    }
    
    if (chat_template_name == "phi3") {
        chat_template = R"(
{% for message in messages %}
{{'<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{'<|assistant|>\n'}}
{% endif %}
)";
    }
    
    if (chat_template_name == "gemma") {
        chat_template = R"(
{% for message in messages %}
{{'<start_of_turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\n' + message['content'] + '<end_of_turn>\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{'<start_of_turn>model\n'}}
{% endif %}
)";
    }
    
    if (chat_template_name == "mistral") {
        chat_template = R"(
{{'<s>'}}
{% for message in messages %}
{% if message['role'] == 'user' %}
{{'[INST] ' + message['content'] + ' [/INST]'}}
{% elif message['role'] == 'system' %}
{{'[INST] ' + message['content'] + ' [/INST]'}}
{% else %}
{{' ' + message['content'] + '</s>'}}
{% endif %}
{% endfor %}
)";
    }
    
    if (chat_template_name == "cohere") {
        chat_template = R"(
{{'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}
{% for message in messages %}
{% if message['role'] == 'system' %}
{{ message['content'] }}
{% endif %}
{% endfor %}
{{'<|END_OF_TURN_TOKEN|>'}}
{% for message in messages %}
{% if message['role'] != 'system' %}
{{'<|START_OF_TURN_TOKEN|><|' + message['role'].upper() + '_TOKEN|>' + message['content'] + '<|END_OF_TURN_TOKEN|>'}}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'}}
{% endif %}
)";
    }
    
    std::string fingerprint;
    long long model_created = 0;
    std::string modelName;
    std::unique_ptr<OgaModel> model;
    std::unique_ptr<OgaTokenizer> tokenizer;
        
    if (model_path.length() != 0) {
        if (fs::exists(model_path)) {
            if (fs::is_directory(model_path)) {
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
        }
    }
    
    std::string embedding_fingerprint;
    long long embedding_model_created = 0;
    std::string embedding_modelName;
    std::unique_ptr<Ort::Session> embeddings_session;
    std::unique_ptr<Ort::Env> embeddings_env;
    size_t num_input_nodes = 0;
    size_t num_output_nodes = 0;
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_shape = {1}; // Batch size 1
    std::vector<const char*> input_names_c_array;
    std::vector<const char*> output_names_c_array;
    std::unique_ptr<Tokenizer> embeddings_tokenizer;
    
    if (embedding_model_path.length() != 0) {
        if (fs::exists(embedding_model_path)) {
            if (fs::is_regular_file(embedding_model_path)) {
                // 1.b Initialize Embedding and Session (Load once)
                std::cerr << "[Embedding] Loading from " << embedding_model_path << std::endl;
                embedding_fingerprint = get_system_fingerprint(embedding_model_path, "directml");
                try {
                    embeddings_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Embeddings");
                    embedding_modelName = get_model_name(embedding_model_path);
                    Ort::SessionOptions session_options;
                    session_options.SetIntraOpNumThreads(1);
                    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                    Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*)session_options, OrtGetApiBase()));
        #ifdef WIN32
                    embeddings_session = std::make_unique<Ort::Session>(*embeddings_env, embedding_model_path_u16.c_str(), session_options);
        #else
                    embeddings_session = std::make_unique<Ort::Session>(*embeddings_env, embedding_model_path.c_str(), session_options);
        #endif
                    num_input_nodes = embeddings_session->GetInputCount();
                    num_output_nodes = embeddings_session->GetOutputCount();
                    for (size_t i = 0; i < num_input_nodes; i++) {
                        auto input_name_ptr = embeddings_session->GetInputNameAllocated(i, allocator);
                        input_node_names.push_back(input_name_ptr.get());
                //        std::cout << "Input " << i << " Name: " << input_name_ptr.get() << std::endl;
                    }
                    for (size_t i = 0; i < num_output_nodes; i++) {
                        auto output_name_ptr = embeddings_session->GetOutputNameAllocated(i, allocator);
                        output_node_names.push_back(output_name_ptr.get());
                //        std::cout << "Input " << i << " Name: " << output_name_ptr.get() << std::endl;
                    }
                    for (const auto& name : input_node_names) {
                        input_names_c_array.push_back(name.c_str());
                    }
                    for (const auto& name : output_node_names) {
                        output_names_c_array.push_back(name.c_str());
                    }
#ifdef WIN32
                    embeddings_tokenizer = LoadTokenizer(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()));
#else
                    embeddings_tokenizer = LoadTokenizer(fs::path(embedding_model_path).parent_path());
#endif
                    embedding_model_created = get_created_timestamp();
                } catch (const std::exception& e) {
                    std::cerr << "Failed to load model: " << e.what() << std::endl;
                    return 1;
                }
            }
        }
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
                
                if(model_created == 0) {
                    throw std::invalid_argument("[Chat] Model not loaded.");
                }
                
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
                                     &is_stream,
                                     tokenizer.get(),
                                     chat_template);
                
                if(is_stream) {
                    std::string req_id = get_openai_style_id();
                    
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
            /*
             The model object
             https://platform.openai.com/docs/api-reference/models/object
             */
            // Create the list wrapper
            Json::Value root(Json::objectValue);
            root["object"] = "list";
            root["data"] = Json::Value(Json::arrayValue);
            // Create the model object
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
            // Serialize
            Json::StreamWriterBuilder writer;
            writer["indentation"] = ""; // Minified JSON
            std::string json_str = Json::writeString(writer, root);
            // Respond
            res.set_content(json_str, "application/json");
            res.status = 200;
        });
        
        // Route: /v1/embeddings
        svr.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/embeddings request received." << std::endl;
            
            try {
                
                if(embedding_model_created == 0) {
                    throw std::invalid_argument("[Embedding] Model not loaded.");
                }
               
                std::string input;
                before_run_embeddings(req.body, input);
                
                std::string response_json;

                if((embeddings_tokenizer != NULL) &&(num_input_nodes > 2)) {
                    /*
                     standard encoder-only model
                     input: input_ids, attention_mask, token_type_ids
                     output: last_hidden_state or logits
                     */
                    std::vector<int> ids = embeddings_tokenizer->Encode(input);
                    response_json = run_embeddings(
                                                   embeddings_session.get(),
                                                   ids, input_names_c_array,
                                                   num_input_nodes,
                                                   output_names_c_array,
                                                   num_output_nodes);
                }else{
                    response_json = run_embeddings_e2e(
                                                       embeddings_session.get(),
                                                       input, input_names_c_array,
                                                       num_input_nodes,
                                                       output_names_c_array,
                                                       num_output_nodes);
                }
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
            
            std::string prompt;
            unsigned int max_tokens = 2048;
            unsigned int top_k = 50;
            double top_p = 0.9;
            double temperature = 0.7;
            unsigned int n = 1;
            bool is_stream = false;
            
            before_run_inference(request_str,
                                 prompt,
                                 &max_tokens,
                                 &top_k,
                                 &top_p,
                                 &temperature,
                                 &n,
                                 &is_stream,
                                 tokenizer.get(),
                                 chat_template);
            
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
