//
//  main.cpp
//  onnx-genai
//
//  Created by miyako on 2025/09/03.
//

#include "onnx-genai.h"

namespace fs = std::filesystem;
using namespace tokenizers;

static std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (!fs) throw std::runtime_error("Could not open file: " + path);
    
    fs.seekg(0, std::ios::end);
    size_t size = fs.tellg();
    std::string data(size, '\0');
    fs.seekg(0, std::ios::beg);
    fs.read(&data[0], size);
    
    return data;
}

static void LoadSpecialTokenIds(const std::string& model_path,
                                RerankingMode ranking_mode,
                                int& cls_id,
                                int& sep_id) {
    
    // 1. Set Defaults based on architecture
    switch (ranking_mode) {
        case RERANKING_MODERNBERT:
            cls_id = 50281;
            sep_id = 50282;
            break;
        case RERANKING_ROBERTA:
            cls_id = 0;
            sep_id = 2;
            break;
        case RERANKING_BERT:
        default:
            cls_id = 101;
            sep_id = 102;
            break;
    }
    
    // 2. Try to read overrides from config.json
    fs::path config_path = fs::path(model_path);
    if (fs::is_directory(config_path)) {
        config_path = config_path / "config.json";
    }
    
    if (fs::exists(config_path) && config_path.extension() == ".json") {
        std::string json = LoadBytesFromFile(config_path.string());
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (reader->parse(json.c_str(), json.c_str() + json.size(), &root, &errors) && root.isObject()) {
            
            // Look for CLS or BOS token
            if (root.isMember("cls_token_id") && root["cls_token_id"].isNumeric()) {
                cls_id = root["cls_token_id"].asInt();
            } else if (root.isMember("bos_token_id") && root["bos_token_id"].isNumeric()) {
                cls_id = root["bos_token_id"].asInt();
            }
            
            // Look for SEP or EOS token
            if (root.isMember("sep_token_id") && root["sep_token_id"].isNumeric()) {
                sep_id = root["sep_token_id"].asInt();
            } else if (root.isMember("eos_token_id") && root["eos_token_id"].isNumeric()) {
                sep_id = root["eos_token_id"].asInt();
            }
        }
    }
    std::cout << "[Tokens] CLS/BOS ID: " << cls_id << " | SEP/EOS ID: " << sep_id << std::endl;
}

static int GetOptimalIntraOpThreads() {
    int threads = 0;

    // --- macOS Implementation ---
    #if defined(__APPLE__)
        int32_t core_count = 0;
        size_t size = sizeof(core_count);
        
        // 1. Try to get "Performance Level 0" cores (P-Cores on Apple Silicon)
        // This is critical for M1/M2/M3 to avoid using slow E-Cores.
        if (sysctlbyname("hw.perflevel0.physicalcpu", &core_count, &size, NULL, 0) == 0) {
            threads = core_count;
        }
        // 2. Fallback: Standard Physical Cores (Intel Mac or if perflevel fails)
        else if (sysctlbyname("hw.physicalcpu", &core_count, &size, NULL, 0) == 0) {
            threads = core_count;
        }
        else {
            // Absolute fallback
            threads = std::thread::hardware_concurrency();
        }

    // --- Windows Implementation ---
    #elif defined(_WIN32)
        // Getting strictly physical cores on Windows is complex (requires iterating SYSTEM_LOGICAL_PROCESSOR_INFORMATION).
        // For a simple implementation, hardware_concurrency (Logical Cores) is often acceptable,
        // but dividing by 2 is a common heuristic for Hyper-threaded Intel/AMD CPUs to estimate physical cores.
        
        unsigned int logical_cores = std::thread::hardware_concurrency();
        // Heuristic: If we have many cores, assume Hyper-threading and divide by 2.
        // Otherwise, use all.
        if (logical_cores > 4) {
            threads = logical_cores / 2;
        } else {
            threads = logical_cores;
        }

    // --- Linux / Generic Implementation ---
    #else
        // Similar heuristic for Linux
        unsigned int logical_cores = std::thread::hardware_concurrency();
        if (logical_cores > 4) {
             threads = logical_cores / 2;
        } else {
             threads = logical_cores;
        }
    #endif

    // Safety clamp: Ensure we have at least 1 thread and not an insane amount (cap at 16 for client devices)
    return std::max(1, std::min(threads, 16));
}

struct RerankResult {
    int index;          // Original index in the document list
    float score;        // Relevance score
    std::string text;   // (Optional) The document text
};

struct RerankItem {
    std::vector<int> ids;
    std::vector<int> type_ids;
};

static // Helper to read the template file from the model directory
std::string LoadChatTemplate(const std::string& model_path) {
    fs::path path(model_path);
    fs::path chat_template_path = path;

    if (fs::is_directory(path)) {
        chat_template_path = path / "chat_template.jinja";
    }
    
    if (fs::exists(chat_template_path) && chat_template_path.extension() == ".jinja") {
//        std::cout << "[Chat] Loading jinja from: " << chat_template_path << std::endl;
        return LoadBytesFromFile(chat_template_path.string());
    }
    
    return "";
}

static // Helper to read the template file from the model directory
RerankingMode LoadRerankingMode(const std::string& model_path) {
    fs::path path(model_path);
    fs::path config_path = path;

    if (fs::is_directory(path)) {
        config_path = path / "config.json";
    }
    
    if (fs::exists(config_path) && config_path.extension() == ".json") {
//        std::cout << "Loading model_type from: " << config_path << std::endl;
        
        std::string json = LoadBytesFromFile(config_path.string());
        
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        bool parse = reader->parse(json.c_str(),
                                   json.c_str() + json.size(),
                                   &root,
                                   &errors);

        if(parse)
        {
            if(root.isObject())
            {
                Json::Value model_type_node = root["model_type"];
                if(model_type_node.isString())
                {
                    std::string model_type = model_type_node.asString();
                    // RERANKING_ROBERTA
                    if(model_type == "xlm-roberta") {
                        std::cout << "[Rerank] model_type: " << model_type << " (roberta)" << std::endl;
                        return RERANKING_ROBERTA;
                    }
                    if(model_type == "roberta") {
                        std::cout << "[Rerank] model_type: " << model_type << std::endl;
                        return RERANKING_ROBERTA;
                    }
                    if(model_type == "camembert") {
                        std::cout << "[Rerank] model_type: " << model_type << " (roberta)" << std::endl;
                        return RERANKING_ROBERTA;
                    }
                    // RERANKING_BERT
                    if(model_type == "bert") {
                        std::cout << "[Rerank] model_type: " << model_type << " (bert)" << std::endl;
                        return RERANKING_BERT;
                    }
                    if(model_type == "mpnet") {
                        std::cout << "[Rerank] model_type: " << model_type << " (bert)" << std::endl;
                        return RERANKING_BERT;
                    }
                    if(model_type == "deberta-v2") {
                        std::cout << "[Rerank] model_type: " << model_type << " (bert)" << std::endl;
                        return RERANKING_BERT;
                    }
                    if(model_type == "modernbert") {
                        std::cout << "[Rerank] model_type: " << model_type << " (bert)" << std::endl;
                        return RERANKING_MODERNBERT;
                    }
                    if(model_type == "qwen3") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    if(model_type == "qwen2") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    if(model_type == "mistral") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    if(model_type == "llama") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    if(model_type == "gemma") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    if(model_type == "gemma2") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    if(model_type == "phi3") {
                        std::cout << "[Rerank] model_type: " << model_type << " (llm)" << std::endl;
                        return RERANKING_LLM;
                    }
                    //
                }
            }
        }
    }
    
    std::cout << "[Rerank] model_type: default (roberta)" << std::endl;
    return RERANKING_ROBERTA;
}

static // Helper to read the template file from the model directory
int LoadMaxPositionEmbeddings(const std::string& model_path) {
    fs::path path(model_path);
    fs::path config_path = path;

    if (fs::is_directory(path)) {
        config_path = path / "config.json";
    }
    
    if (fs::exists(config_path) && config_path.extension() == ".json") {
//        std::cout << "Loading max_position_embeddings from: " << config_path << std::endl;
        
        std::string json = LoadBytesFromFile(config_path.string());
        
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        bool parse = reader->parse(json.c_str(),
                                   json.c_str() + json.size(),
                                   &root,
                                   &errors);
        
        if(parse)
        {
            if(root.isObject())
            {
                Json::Value max_position_embeddings_node = root["max_position_embeddings"];
                if(max_position_embeddings_node.isNumeric())
                {
                    return  max_position_embeddings_node.asInt();
                }
            }
        }
    }
    
    return 512;
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
//        std::cout << "Loading HF Tokenizer from: " << json_path << std::endl;
        std::string blob = LoadBytesFromFile(json_path.string());
        return Tokenizer::FromBlobJSON(blob);
    }
    
    // 3. Fallback to SentencePiece
    if (fs::exists(model_file_path) && model_file_path.extension() == ".model") {
//        std::cout << "Loading SentencePiece from: " << model_file_path << std::endl;
        std::string blob = LoadBytesFromFile(model_file_path.string());
        return Tokenizer::FromBlobSentencePiece(blob);
    }

    return nullptr;
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
    for (long i = 0; i < batch_size; ++i) {
        
        // --- Step A: Optimized Mean Pooling (Inlined) ---
        // We write directly into out.row(i) to avoid creating temporary VectorXf objects.
        
        const auto& hidden = hidden_batch[i];
        const auto& mask = mask_batch[i];

        // Convert mask to float for calculation
        Eigen::VectorXf mask_f = mask.cast<float>();
        float count = mask_f.sum();

        if (count > 1e-9f) { // Use a small epsilon instead of 0.0f
            // Matrix Mult: [1, seq] * [seq, dim] -> [1, dim]
            out.row(i) = mask_f.transpose() * hidden;
            out.row(i) /= count; // Average
            
            // MATH IMPROVEMENT: Use Eigen's in-place normalization
            // This is generally safer and cleaner.
            out.row(i).normalize();
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
    fprintf(stderr, " -%c path     : %s\n", 'r' , "reranker model");
    fprintf(stderr, " -%c          : %s\n", 'j' , "chat template from stdin");
    fprintf(stderr, " -%c path     : %s\n", 't' , "chat template");
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
#define ARGS (OPTARG_T)L"m:e:r:i:o:sp:jt:bcld-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:e:r:i:o:sp:jt:bcld-h"
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
    return path.filename().string();
}

// Simple stable FNV-1a hash implementation
static std::string get_system_fingerprint(const std::string& model_path, const std::string& provider) {
    std::string identifier = model_path + "_" + provider;
    uint64_t hash = 14695981039346656037ULL;
    for (char c : identifier) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    std::stringstream ss;
    ss << "fp_" << std::hex << hash;
    return ss.str();
}

static std::string get_openai_style_id() {
    const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    
    // Initialize once per thread
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, max_index - 1);
    
    std::string id = "chatcmpl-";
    id.reserve(29 + 9); // reserve space to avoid reallocation
    for (int i = 0; i < 29; ++i) {
        id += charset[dis(gen)];
    }
    return id;
}

#pragma mark -

static void parse_request_reranking(const std::string &json,
                                     std::string &query,
                                     int *top_n,
                                     std::vector<std::string> &documents
                                     ) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    
    if(parse)
    {
        if(root.isObject())
        {
            Json::Value query_node = root["query"];
            if(query_node.isString())
            {
                query = query_node.asString();
            }
            Json::Value top_n_node = root["top_n"];
            if(top_n_node.isNumeric())
            {
                *top_n = top_n_node.asInt();
            }
            
            Json::Value documents_node = root["documents"];
            if(documents_node.isArray())
            {
                for(Json::Value::const_iterator it = documents_node.begin() ; it != documents_node.end() ; it++)
                {
                    if(it->isString())
                    {
                        std::string document = it->asString();
                        documents.push_back(document);
                    }
                }
            }
        }
    }
}

static void parse_request_contextualized_embeddings(const std::string &json,
                                     std::vector<std::string> &inputs) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    
    if(parse && root.isObject())
    {
        // Voyage AI uses "inputs" (plural)
        Json::Value inputs_node = root["inputs"];
        
        // fallback for 4D AI Kit which uses "inout" (singular)
        inputs_node = inputs_node.isArray() ? inputs_node : root["input"];
        
        if(inputs_node.isArray())
        {
            // Iterate over documents (each document is an array of chunks)
            for (Json::Value::const_iterator it_doc = inputs_node.begin(); it_doc != inputs_node.end(); ++it_doc)
            {
                const Json::Value& chunk_array = *it_doc;
                if(chunk_array.isArray())
                {
                    // 1. Reconstruct the full document by concatenating its chunks
                    std::string full_document;
                    for (Json::Value::const_iterator it_chunk = chunk_array.begin(); it_chunk != chunk_array.end(); ++it_chunk)
                    {
                        if(it_chunk->isString()) {
                            full_document += it_chunk->asString();
                        }
                    }
                    
                    // 2. Flatten the request: create a contextualized input for each chunk
                    for (Json::Value::const_iterator it_chunk = chunk_array.begin(); it_chunk != chunk_array.end(); ++it_chunk)
                    {
                        if(it_chunk->isString()) {
                            std::string chunk = it_chunk->asString();
                            // Prepend the reconstructed document context to the specific chunk.
                            // This allows standard ONNX models to approximate Voyage's context-awareness.
                            inputs.push_back(full_document + "\n\n" + chunk);
                        }
                    }
                }
            }
        }
    }
}

static void parse_request_embeddings(const std::string &json,
                                     std::vector<std::string> &inputs) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    
    if(parse)
    {
        if(root.isObject())
        {
            Json::Value input_node = root["input"];
            if(input_node.isString())
            {
                inputs.push_back(input_node.asString());
            }
            if(input_node.isArray())
            {
                for (Json::ValueIterator i = input_node.begin(); i != input_node.end(); ++i)
                {
                    const auto& node = *i;
                    if(node.isString())
                    {
                        inputs.push_back(node.asString());
                    }
                }
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
                          double *repetition_penalty,
                          unsigned int *n,
                          bool *is_stream,
                          OgaTokenizer* tokenizer,
                          std::string& chat_template,
                          std::string& guidance_string_type,
                          std::string& guidance_string) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
        
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
            Json::Value repetition_penalty_node = root["repetition_penalty"];
            if(repetition_penalty_node.isNumeric())
            {
                *repetition_penalty = repetition_penalty_node.asDouble();
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
            Json::Value response_format_node = root["response_format"];
            if(response_format_node.isObject())
            {
                Json::Value response_format_type_node = response_format_node["type"];
                if(response_format_type_node.isString())
                {
                    std::string response_format_type = response_format_type_node.asString();
                    if(response_format_type == "json_schema") {
                        Json::Value json_schema_node = response_format_node["json_schema"];
                        if(json_schema_node.isObject())
                        {
                            Json::Value schema_node = json_schema_node["schema"];
                            if(schema_node.isObject())
                            {
                                Json::StreamWriterBuilder writer;
                                writer["indentation"] = "";
                                guidance_string = Json::writeString(writer, schema_node);
                                guidance_string_type = "json_schema";
                            }
                        }
                    }
                    if(response_format_type == "regex") {
                        Json::Value regex_node = response_format_node["regex"];
                        if(regex_node.isString())
                        {
                            guidance_string = regex_node.asString();
                            guidance_string_type = "regex";
                        }
                    }
                    if(response_format_type == "lark_grammar") {
                        Json::Value lark_grammar_node = response_format_node["lark_grammar"];
                        if(lark_grammar_node.isString())
                        {
                            guidance_string = lark_grammar_node.asString();
                            guidance_string_type = "lark_grammar";
                        }
                    }
                }
            }
        }
    }
}

#pragma mark -

static void before_run_reranking(
                                 const std::string& request_body,
                                 std::string &query,
                                 int *top_n,
                                 std::vector<std::string> &documents
                                 ) {
    parse_request_reranking(request_body, query, top_n, documents);
}

static void before_run_contextualized_embeddings(
                                  const std::string& request_body,
                                  std::vector<std::string> &inputs
                                  ) {
    parse_request_contextualized_embeddings(request_body, inputs);
}

static void before_run_embeddings(
                                  const std::string& request_body,
                                  std::vector<std::string> &inputs
                                  ) {
    parse_request_embeddings(request_body, inputs);
}

static void before_run_inference(
                                 const std::string& request_body,
                                 std::string &prompt,
                                 unsigned int *max_tokens,
                                 unsigned int *top_k,
                                 double *top_p,
                                 double *temperature,
                                 double *repetition_penalty,
                                 unsigned int *n,
                                 bool *is_stream,
                                 OgaTokenizer* tokenizer,
                                 std::string& chat_template,
                                 std::string& guidance_string_type,
                                 std::string& guidance_string) {
    
    parse_request(request_body, prompt, max_tokens, top_k, top_p, temperature, repetition_penalty, n, is_stream, tokenizer, chat_template, guidance_string_type, guidance_string);
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
                                 double repetition_penalty,
                                 unsigned int n,
                                 std::string prompt,
                                 std::string guidance_string_type,
                                 std::string guidance_string
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
    params->SetSearchOption("repetition_penalty", repetition_penalty);
    params->SetSearchOption("num_return_sequences", n);
    
    if(guidance_string_type != ""){
        params->SetGuidance(guidance_string_type.c_str(), guidance_string.c_str());
    }
#if TOKEN_BACKSTOP
    int32_t chat_end_id = tokenizer->ToTokenId("<|im_end|>");
    int32_t file_end_id = tokenizer->ToTokenId("<|endoftext|>");
    int32_t chat_start_id = tokenizer->ToTokenId("<|im_start|>");
    int32_t head_start_id = tokenizer->ToTokenId("<|start_header_id|>");
    int32_t pad_id = tokenizer->ToTokenId("<pad>");
    int32_t bos_id = tokenizer->ToTokenId("<bos>");
    int32_t turn_start_id = tokenizer->ToTokenId("<start_of_turn>");
    int32_t turn_end_id = tokenizer->ToTokenId("<end_of_turn>");
    int32_t end_id = tokenizer->ToTokenId("<|end|>");
    
    std::unordered_set<int32_t> stop_tokens = {
        chat_end_id,
        file_end_id,
        chat_start_id,
        head_start_id,
        pad_id,
        bos_id,
        turn_start_id,
        turn_end_id,
        end_id};
#endif
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
#if TOKEN_BACKSTOP
            if (stop_tokens.count(new_token)) {
                // We hit one of our stop tokens!
                continue;
            }
#endif
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
                                       const std::string& fingerprint,
                                       const std::string& content,
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
                                 double repetition_penalty,
                                 unsigned int n,
                                 std::string prompt,
                                 std::string guidance_string_type,
                                 std::string guidance_string,
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
    params->SetSearchOption("repetition_penalty", repetition_penalty);
    params->SetSearchOption("num_return_sequences", n);
    
    if(guidance_string_type != ""){
        params->SetGuidance(guidance_string_type.c_str(), guidance_string.c_str());
    }
#if TOKEN_BACKSTOP
    int32_t chat_end_id = tokenizer->ToTokenId("<|im_end|>");
    int32_t file_end_id = tokenizer->ToTokenId("<|endoftext|>");
    int32_t chat_start_id = tokenizer->ToTokenId("<|im_start|>");
    int32_t head_start_id = tokenizer->ToTokenId("<|start_header_id|>");
    int32_t pad_id = tokenizer->ToTokenId("<pad>");
    int32_t bos_id = tokenizer->ToTokenId("<bos>");
    int32_t turn_start_id = tokenizer->ToTokenId("<start_of_turn>");
    int32_t turn_end_id = tokenizer->ToTokenId("<end_of_turn>");
    int32_t end_id = tokenizer->ToTokenId("<|end|>");

    std::unordered_set<int32_t> stop_tokens = {
        chat_end_id,
        file_end_id,
        chat_start_id,
        head_start_id,
        pad_id,
        bos_id,
        turn_start_id,
        turn_end_id,
        end_id};
#endif
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
#if TOKEN_BACKSTOP
            if (stop_tokens.count(new_token)) {
                // We hit one of our stop tokens!
                continue;
            }
#endif
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

static std::vector<std::vector<float>> last_token_pooling_batch(
    std::vector<Ort::Value>& outputs,
    const std::vector<int64_t>& attention_mask,
    int batch_size, int max_seq_len)
{
    std::vector<std::vector<float>> batch_embeddings;
    if (outputs.empty()) return batch_embeddings;
        
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() <= 2) return batch_embeddings;
    
    int64_t hidden_size = shape[2];
    float* floatarr = outputs[0].GetTensorMutableData<float>();
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            raw_matrix(floatarr + (b * max_seq_len * hidden_size), max_seq_len, hidden_size);
        
        int last_token_index = 0;
        for (int i = 0; i < max_seq_len; ++i) {
            if (attention_mask[b * max_seq_len + i] == 1) {
                last_token_index = i;
            } else {
                break; // Padding found
            }
        }
        
        Eigen::VectorXf final_embedding = raw_matrix.row(last_token_index).normalized();
        batch_embeddings.push_back(std::vector<float>(final_embedding.data(), final_embedding.data() + final_embedding.size()));
    }
    return batch_embeddings;
}

// ColBERT returns a list of embeddings per token, so we build the JSON directly.
static std::string colbert_pooling_batch_json(
    std::vector<Ort::Value>& outputs,
    const std::vector<int64_t>& attention_mask,
    int batch_size, int max_seq_len)
{
    Json::Value rootNode(Json::objectValue);
    Json::Value listNode(Json::arrayValue);
    
    if(!outputs.empty()) {
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if(shape.size() > 2) {
            int64_t hidden_size = shape[2];
            float* floatarr = outputs[0].GetTensorMutableData<float>();
            
            for (int b = 0; b < batch_size; ++b) {
                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    raw_matrix(floatarr + (b * max_seq_len * hidden_size), max_seq_len, hidden_size);
                
                Eigen::MatrixXf normalized_matrix = raw_matrix.rowwise().normalized();
                
                Json::Value dataNode = Json::objectValue;
                dataNode["object"] = "embedding";
                dataNode["index"] = b;
                
                Json::Value tokensArray(Json::arrayValue);
                for (int i = 0; i < max_seq_len; ++i) {
                    if (attention_mask[b * max_seq_len + i] == 0) continue; // Skip padding
                    
                    Json::Value tokenEmbedding(Json::arrayValue);
                    for (int j = 0; j < hidden_size; ++j) {
                        tokenEmbedding.append(normalized_matrix(i, j));
                    }
                    tokensArray.append(tokenEmbedding);
                }
                dataNode["embedding"] = tokensArray;
                listNode.append(dataNode);
            }
        }
    }
    rootNode["data"] = listNode;
    rootNode["object"] = "list";
    
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

static std::vector<std::vector<float>> cls_pooling_batch(
    std::vector<Ort::Value>& outputs,
    const std::vector<int64_t>& attention_mask, // Not strictly needed for CLS, but kept for signature consistency
    int batch_size, int max_seq_len)
{
    std::vector<std::vector<float>> batch_embeddings;
    if (outputs.empty()) return batch_embeddings;
     
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() <= 2) return batch_embeddings;
    
    int64_t hidden_size = shape[2];
    float* floatarr = outputs[0].GetTensorMutableData<float>();
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            raw_matrix(floatarr + (b * max_seq_len * hidden_size), max_seq_len, hidden_size);
        
        Eigen::VectorXf cls_vec = raw_matrix.row(0);
        Eigen::VectorXf final_embedding = l2_normalize(cls_vec);
        batch_embeddings.push_back(std::vector<float>(final_embedding.data(), final_embedding.data() + final_embedding.size()));
    }
    return batch_embeddings;
}

static std::vector<std::vector<float>> mean_pooling_batch(
    std::vector<Ort::Value>& outputs,
    const std::vector<int64_t>& attention_mask,
    int batch_size, int max_seq_len)
{
    std::vector<std::vector<float>> batch_embeddings;
    if (outputs.empty()) return batch_embeddings;
        
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() <= 2) return batch_embeddings;
    
    int64_t hidden_size = shape[2];
    float* floatarr = outputs[0].GetTensorMutableData<float>();
    
    std::vector<Eigen::MatrixXf> hidden_batch_vec;
    std::vector<Eigen::VectorXi> mask_batch_vec;
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mapped_hidden(floatarr + (b * max_seq_len * hidden_size), max_seq_len, hidden_size);
        hidden_batch_vec.push_back(mapped_hidden);
        
        Eigen::VectorXi mask_vec(max_seq_len);
        for(int i = 0; i < max_seq_len; ++i) {
            mask_vec(i) = (int)attention_mask[b * max_seq_len + i];
        }
        mask_batch_vec.push_back(mask_vec);
    }
    
    // Process the entire batch using OpenMP optimized function
    Eigen::MatrixXf pooled = mean_pool_batch(hidden_batch_vec, mask_batch_vec); // Returns [Batch, Hidden]
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::VectorXf final_embedding = l2_normalize(pooled.row(b));
        batch_embeddings.push_back(std::vector<float>(final_embedding.data(), final_embedding.data() + final_embedding.size()));
    }
    return batch_embeddings;
}

static std::string run_reranking(
    Ort::Session *session,
    std::vector<RerankItem>& items,
    int max_position_embeddings,
    int top_n,
    std::vector<const char*>& input_names_c_array,
    size_t num_input_nodes,
    std::vector<const char*>& output_names_c_array,
    size_t num_output_nodes,
    RerankingMode ranking_mode)
{
    if (items.empty()) {
        return "{\"object\":\"list\",\"results\":[]}";
    }

    try {
        int batch_size = (int)items.size();
        int max_seq_len = 0;

        // 1. Find max length in batch
        for (auto& item : items) {
            if (item.ids.size() > max_position_embeddings) {
                item.ids.resize(max_position_embeddings);
                if (!item.type_ids.empty()) item.type_ids.resize(max_position_embeddings);
            }
            if ((int)item.ids.size() > max_seq_len) {
                max_seq_len = (int)item.ids.size();
            }
        }

        // 2. Allocate flat memory (Zero-initialized for padding)
        size_t total_elements = (size_t)batch_size * max_seq_len;
        std::vector<int64_t> flat_ids(total_elements, 0);
        std::vector<int64_t> flat_mask(total_elements, 0);
        std::vector<int64_t> flat_type(total_elements, 0);

        // 3. Fill the flat arrays
        for (int b = 0; b < batch_size; ++b) {
            int seq_len = (int)items[b].ids.size();
            for (int i = 0; i < seq_len; ++i) {
                size_t idx = (size_t)(b * max_seq_len + i);
                flat_ids[idx] = items[b].ids[i];
                flat_mask[idx] = 1;
                if (i < items[b].type_ids.size()) {
                    flat_type[idx] = items[b].type_ids[i];
                }
            }
        }

        // 4. Create Tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = { batch_size, max_seq_len };
        std::vector<Ort::Value> input_tensors;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_ids.data(), flat_ids.size(), input_dims.data(), input_dims.size()));

        if (num_input_nodes > 1) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_mask.data(), flat_mask.size(), input_dims.data(), input_dims.size()));
            
            if (num_input_nodes > 2) {
                input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info, flat_type.data(), flat_type.size(), input_dims.data(), input_dims.size()));
            }
        }

        // 5. Run Batched Inference (1 Call!)
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names_c_array.data(),
            input_tensors.data(),
            num_input_nodes,
            output_names_c_array.data(),
            num_output_nodes
        );

        // 6. Extract Outputs and Vectorized Math
        float* float_data = outputs.front().GetTensorMutableData<float>();
        auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        int output_dim = (int)shape.back(); // Usually 1 or 2

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            logits_mat(float_data, batch_size, output_dim);
        
        Eigen::ArrayXf final_scores(batch_size);

        if (output_dim == 2) {
            final_scores = (1.0f + (logits_mat.col(0) - logits_mat.col(1)).array().exp()).inverse();
        } else {
            final_scores = (1.0f + (-logits_mat.col(0)).array().exp()).inverse();
        }

        // 7. Sort & Build JSON
        std::vector<RerankResult> results;
        results.reserve(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            results.push_back({b, final_scores[b]});
        }

        auto sorter = [](const RerankResult& a, const RerankResult& b) {
            return a.score > b.score;
        };
        
        if (top_n > 0 && top_n < batch_size) {
            std::partial_sort(results.begin(), results.begin() + top_n, results.end(), sorter);
            results.resize(top_n);
        } else {
            std::sort(results.begin(), results.end(), sorter);
        }
        
        Json::Value rootNode(Json::objectValue);
        Json::Value listNode(Json::arrayValue);
        for (const auto& result : results) {
            Json::Value dataNode = Json::objectValue;
            dataNode["index"] = result.index;
            dataNode["relevance_score"] = result.score;
            listNode.append(dataNode);
        }

        rootNode["results"] = listNode;
        rootNode["object"] = "list";
        
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Json::writeString(writer, rootNode);

    } catch (const std::exception& e) {
        throw;
    }
}

static std::string run_embeddings(
                                  Ort::Session *session,
                                  std::vector<std::string> &inputs,
                                  int max_position_embeddings,
                                  std::vector<const char*>& input_names_c_array,
                                  size_t num_input_nodes,
                                  std::vector<const char*>& output_names_c_array,
                                  size_t num_output_nodes,
                                  Tokenizer* tokenizer,
                                  PoolingMode pooling_mode,
                                  int cls_id,
                                  int sep_id)
{
    if (tokenizer == nullptr || inputs.empty()) {
        return "{\"object\":\"list\",\"data\":[]}";
    }

    try {
        int batch_size = (int)inputs.size();
        int max_seq_len = 0;
        std::vector<std::vector<int>> tokenized_inputs;
        tokenized_inputs.reserve(batch_size);

        // 1. Tokenize all and find the max length for padding
        for (const auto& input : inputs) {
            std::vector<int> ids = tokenizer->Encode(input);

            ids.insert(ids.begin(), cls_id);
            ids.push_back(sep_id);

            // Handle Truncation safely
            if (ids.size() > static_cast<size_t>(max_position_embeddings)) {
                ids.resize(max_position_embeddings - 1);
                ids.push_back(sep_id); // Ensure it always ends with the correct token
            }

            if ((int)ids.size() > max_seq_len) {
                max_seq_len = (int)ids.size();
            }
            tokenized_inputs.push_back(std::move(ids));
        }

        // 2. Allocate flat memory for tensors (Zero initialized for padding)
        size_t total_elements = (size_t)batch_size * max_seq_len;
        std::vector<int64_t> flat_input_ids(total_elements, 0);
        std::vector<int64_t> flat_attention_mask(total_elements, 0);
        std::vector<int64_t> flat_token_type_ids(total_elements, 0);

        // 3. Fill the flat arrays
        for (int b = 0; b < batch_size; ++b) {
            int seq_len = (int)tokenized_inputs[b].size();
            for (int i = 0; i < seq_len; ++i) {
                size_t idx = (size_t)(b * max_seq_len + i);
                flat_input_ids[idx] = tokenized_inputs[b][i];
                flat_attention_mask[idx] = 1; // 1 for real tokens, pad remains 0
                // flat_token_type_ids remains 0
            }
        }

        // 4. Create Tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = { batch_size, max_seq_len };
        std::vector<Ort::Value> input_tensors;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_input_ids.data(), flat_input_ids.size(), input_dims.data(), input_dims.size()));
        
        if (num_input_nodes > 1) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_attention_mask.data(), flat_attention_mask.size(), input_dims.data(), input_dims.size()));
        }
        if (num_input_nodes > 2) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_token_type_ids.data(), flat_token_type_ids.size(), input_dims.data(), input_dims.size()));
        }

        // 5. Run Batched Inference (1 Call to session->Run!)
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names_c_array.data(),
            input_tensors.data(),
            num_input_nodes,
            output_names_c_array.data(),
            num_output_nodes
        );

        // 6. Pooling & Build JSON
        if (pooling_mode == POOLING_COLBERT) {
            // Colbert requires specialized nested JSON construction
            return colbert_pooling_batch_json(outputs, flat_attention_mask, batch_size, max_seq_len);
        }
        
        std::vector<std::vector<float>> batch_embeddings;
        switch (pooling_mode) {
            case POOLING_CLS:
                batch_embeddings = cls_pooling_batch(outputs, flat_attention_mask, batch_size, max_seq_len);
                break;
            case POOLING_LAST_TOKEN:
                batch_embeddings = last_token_pooling_batch(outputs, flat_attention_mask, batch_size, max_seq_len);
                break;
            case POOLING_MEAN:
            default:
                batch_embeddings = mean_pooling_batch(outputs, flat_attention_mask, batch_size, max_seq_len);
                break;
        }

        Json::Value rootNode(Json::objectValue);
        Json::Value listNode(Json::arrayValue);
        
        for (int b = 0; b < batch_size; ++b) {
            Json::Value dataNode = Json::objectValue;
            Json::Value embeddingsNode(Json::arrayValue);
            for (float val : batch_embeddings[b]) {
                embeddingsNode.append(val);
            }
            dataNode["object"] = "embedding";
            dataNode["embedding"] = embeddingsNode;
            dataNode["index"] = b;
            listNode.append(dataNode);
        }
        
        rootNode["data"] = listNode;
        rootNode["object"] = "list";
        
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Json::writeString(writer, rootNode);

    } catch (const std::exception& e) {
        throw; // Controller handles the JSON error formatting
    }
}

static std::string run_embeddings_e2e(
                                      Ort::Session *session,
                                      std::vector<std::string> &inputs,
                                      std::vector<const char*>&  input_names_c_array,
                                      size_t num_input_nodes,
                                      std::vector<const char*>&   output_names_c_array,
                                      size_t num_output_nodes) {
        
    // Build JSON Response
    Json::Value rootNode(Json::objectValue);
    Json::Value listNode(Json::arrayValue);
    int b = 0;
    for (const auto& input : inputs) {
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
            api.ReleaseValue(raw_tensor_ptr);
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
            
            auto shape = output_info.GetShape();
            if(shape.size() > 0) {
                int64_t embedding_dim = shape[1];
                // Create the std::vector
                std::vector<float> embeddings(floatarr, floatarr + embedding_dim);
                
                Json::Value dataNode = Json::objectValue;
                dataNode["object"] = "embedding";
                Json::Value embeddingsNode(Json::arrayValue);
                for (float val : embeddings) {
                    embeddingsNode.append(val);
                }
                dataNode["embedding"] = embeddingsNode;
                dataNode["index"] = b;
                Json::Value listNode = Json::arrayValue;
                listNode.append(dataNode);
            }
        }
        b++;
    }
    
    rootNode["data"] = listNode;
    rootNode["object"] = "list";
    
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, rootNode);
}

static std::string run_colbert_reranking(
                                         Ort::Session *session,
                                         const std::string& query,
                                         const std::vector<std::string>& documents,
                                         Tokenizer* tokenizer,
                                         int max_position_embeddings,
                                         int top_n,
                                         std::vector<const char*>& input_names_c_array,
                                         size_t num_input_nodes,
                                         std::vector<const char*>& output_names_c_array,
                                         size_t num_output_nodes,
                                         RerankingMode ranking_mode,
                                         int cls_id,
                                         int sep_id)
{
    if (documents.empty()) {
        return "{\"object\":\"list\",\"results\":[]}";
    }

    try {
        std::vector<std::string> inputs;
        inputs.push_back(query);
        inputs.insert(inputs.end(), documents.begin(), documents.end());

        int batch_size = (int)inputs.size();
        int max_seq_len = 0;
        std::vector<std::vector<int>> tokenized_inputs;
        tokenized_inputs.reserve(batch_size);

        // 1. Tokenize query and all documents
        for (int i = 0; i < batch_size; ++i) {
            std::vector<int> raw_ids = tokenizer->Encode(inputs[i]);
            std::vector<int> ids;
            ids.reserve(raw_ids.size() + 2);

            switch (ranking_mode) {
                case RERANKING_MODERNBERT:
                case RERANKING_BERT:
                case RERANKING_ROBERTA:
                    ids.push_back(cls_id);
                    ids.insert(ids.end(), raw_ids.begin(), raw_ids.end());
                    ids.push_back(sep_id);
                    break;
                default:
                    ids = raw_ids;
                    break;
            }
            
            if (ids.size() > static_cast<size_t>(max_position_embeddings)) {
                ids.resize(max_position_embeddings - 1);
                if (ranking_mode == RERANKING_BERT) ids.push_back(102);
                else if (ranking_mode == RERANKING_ROBERTA) ids.push_back(2);
            }
            
            if ((int)ids.size() > max_seq_len) {
                max_seq_len = (int)ids.size();
            }
            tokenized_inputs.push_back(std::move(ids));
        }

        // 2. Allocate flat memory (Zero-initialized for padding)
        size_t total_elements = (size_t)batch_size * max_seq_len;
        std::vector<int64_t> flat_input_ids(total_elements, 0);
        std::vector<int64_t> flat_attention_mask(total_elements, 0);
        std::vector<int64_t> flat_token_type_ids(total_elements, 0);

        // 3. Fill the flat arrays
        for (int b = 0; b < batch_size; ++b) {
            int seq_len = (int)tokenized_inputs[b].size();
            for (int i = 0; i < seq_len; ++i) {
                size_t idx = (size_t)(b * max_seq_len + i);
                flat_input_ids[idx] = tokenized_inputs[b][i];
                flat_attention_mask[idx] = 1;
            }
        }

        // 4. Create Tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = { batch_size, max_seq_len };
        std::vector<Ort::Value> input_tensors;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_input_ids.data(), flat_input_ids.size(), input_dims.data(), input_dims.size()));
        
        if (num_input_nodes > 1) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_attention_mask.data(), flat_attention_mask.size(), input_dims.data(), input_dims.size()));
        }
        if (num_input_nodes > 2) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_token_type_ids.data(), flat_token_type_ids.size(), input_dims.data(), input_dims.size()));
        }

        // 5. Run Batched Inference
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names_c_array.data(),
            input_tensors.data(),
            num_input_nodes,
            output_names_c_array.data(),
            num_output_nodes
        );

        float* data = outputs.front().GetTensorMutableData<float>();
        auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        
        if (shape.size() < 3) {
            throw std::runtime_error("ColBERT reranking requires a 3D tensor output [batch, seq_len, hidden_size].");
        }
        int64_t hidden_size = shape[2];

        // 6. Process Query Embeddings (Batch Index 0)
        int q_len = 0;
        for(int i = 0; i < max_seq_len; ++i) {
            if(flat_attention_mask[i] == 1) q_len++;
        }
        
        Eigen::MatrixXf Q_valid(q_len, hidden_size);
        if (q_len > 0) {
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                raw_Q(data, max_seq_len, hidden_size);
            int q_idx = 0;
            for(int i = 0; i < max_seq_len; ++i) {
                if(flat_attention_mask[i] == 1) {
                    // ColBERT requires L2 Normalized token embeddings
                    Q_valid.row(q_idx++) = raw_Q.row(i).normalized();
                }
            }
        }

        // 7. Process Document Embeddings & MaxSim Scoring
        std::vector<RerankResult> results;
        results.reserve(documents.size());

        for (int b = 1; b < batch_size; ++b) {
            int d_len = 0;
            for(int i = 0; i < max_seq_len; ++i) {
                if(flat_attention_mask[b * max_seq_len + i] == 1) d_len++;
            }

            if (q_len == 0 || d_len == 0) {
                results.push_back({b - 1, 0.0f});
                continue;
            }

            Eigen::MatrixXf D_valid(d_len, hidden_size);
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                raw_D(data + b * max_seq_len * hidden_size, max_seq_len, hidden_size);
            
            int d_idx = 0;
            for(int i = 0; i < max_seq_len; ++i) {
                if(flat_attention_mask[b * max_seq_len + i] == 1) {
                    D_valid.row(d_idx++) = raw_D.row(i).normalized();
                }
            }

            // MaxSim math: Query tokens (rows) x Doc tokens (cols) -> Resulting in [q_len, d_len]
            Eigen::MatrixXf Sim = Q_valid * D_valid.transpose();
            
            // For each query token, find max similarity across doc tokens, then sum for total score
            float score = Sim.rowwise().maxCoeff().sum();
            results.push_back({b - 1, score});
        }

        // 8. Sort and Build JSON
        auto sorter = [](const RerankResult& a, const RerankResult& b) {
            return a.score > b.score;
        };
        
        if (top_n > 0 && top_n < (int)results.size()) {
            std::partial_sort(results.begin(), results.begin() + top_n, results.end(), sorter);
            results.resize(top_n);
        } else {
            std::sort(results.begin(), results.end(), sorter);
        }

        Json::Value rootNode(Json::objectValue);
        Json::Value listNode(Json::arrayValue);
        for (const auto& result : results) {
            Json::Value dataNode = Json::objectValue;
            dataNode["index"] = result.index;
            dataNode["relevance_score"] = result.score;
            listNode.append(dataNode);
        }

        rootNode["results"] = listNode;
        rootNode["object"] = "list";
        
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Json::writeString(writer, rootNode);

    } catch (const std::exception& e) {
        throw;
    }
}

#pragma mark -

int main(int argc, OPTARG_T argv[]) {
    
#ifdef WIN32
    std::wstring model_path_u16;
    std::wstring embedding_model_path_u16;
    std::wstring reranker_model_path_u16;
#endif
    std::string model_path;           // -m
    std::string embedding_model_path; // -e
    std::string reranker_model_path;  // -r
    std::string chat_template;        // -j
    OPTARG_T input_path  = NULL;      // -i
    OPTARG_T output_path = NULL;      // -o
    OPTARG_T chat_template_path = NULL;
        
    PoolingMode pooling_mode = POOLING_MEAN;
    
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
            case 'r':
#ifdef WIN32
                reranker_model_path_u16 = optarg;
                reranker_model_path = wchar_to_utf8(reranker_model_path_u16.c_str());
#else
                reranker_model_path = optarg;
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
            case 'b':
                pooling_mode = POOLING_COLBERT;
                break;
            case 'c':
                pooling_mode = POOLING_CLS;
                break;
            case 'l':
                pooling_mode = POOLING_LAST_TOKEN;
                break;
            case 'd':
                pooling_mode = POOLING_E2E;
                break;
            case 'h':
#ifdef WIN32
                host = wchar_to_utf8(optarg);
#else
                host = optarg;
#endif
                break;
            case 'j':
            case '-':
            {
                // Only relevant for CLI mode
                std::vector<uint8_t> buf(BUFLEN);
                size_t s;
                while ((s = fread(buf.data(), 1, buf.size(), stdin)) > 0) {
                    cli_request_json.insert(cli_request_json.end(), buf.begin(), buf.begin() + s);
                }
                if(ch == 'j') {
                    chat_template = std::string((const char *)cli_request_json.data(), cli_request_json.size());
                }
            }
                break;
            case 't':
            {
                chat_template_path = optarg;
                if (chat_template_path != NULL){
                    FILE *f = _fopen(chat_template_path, _rb);
                    if(f) {
                        std::vector<unsigned char> chat_template_string(0);
                        fseek(f, 0, SEEK_END);
                        size_t len = (size_t)ftell(f);
                        fseek(f, 0, SEEK_SET);
                        chat_template_string.resize(len);
                        fread(chat_template_string.data(), 1, chat_template_string.size(), f);
                        fclose(f);
                        chat_template = std::string((const char *)chat_template_string.data(), chat_template_string.size());
                    }
                }
            }
                break;
            default:
                usage();
                break;
        }
    }
    
    int intra_op_threads = GetOptimalIntraOpThreads();
    std::cout << "Detected " << intra_op_threads << " Intra-Op threads." << std::endl;
    
    std::string fingerprint;
    long long model_created = 0;
    std::string modelName;
    std::unique_ptr<OgaModel> model;
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaConfig> config;
    
    if (model_path.length() != 0) {
        if (fs::exists(model_path)) {
            if (fs::is_directory(model_path)) {
                
                model_path = fs::path(model_path).lexically_normal().string();
                                                
                // 1.a Initialize Model and Tokenizer (Load once)
                std::cerr << "[Chat] Loading from " << model_path << std::endl;
                modelName = get_model_name(model_path);

                // We determine the provider dynamically now, so we track it for the fingerprint
                std::string active_provider = "CPU";
                                
                try {
                    // 1. Create the Config Object
                    config = OgaConfig::Create(model_path.c_str());
                    config->ClearProviders();
                    
                    // 2. Dynamic Provider Loading Logic
#if defined(_WIN32)
//                    try{
//                        config->AppendProvider("DML");
//                        active_provider = "DML";
//                    } catch (const std::exception& e) {
//                        std::cerr << "Failed append provider: " << e.what() << std::endl;
//                    }
//https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options
#elif defined(__APPLE__)
//                    try{
//                        config->AppendProvider("CoreML");
//                        active_provider = "CoreML";
//                    } catch (const std::exception& e) {
//                        std::cerr << "Failed to load model: " << e.what() << std::endl;
//                    }
#endif
                    // 3. Update Fingerprint with actual provider used
                    fingerprint = get_system_fingerprint(model_path, active_provider);
                    
                    // 4. Create Model from the Config
//#if defined(_WIN32)
//                    try{
//                        model = OgaModel::Create(*config);
//                    } catch (const std::exception& e) {
//                        std::cerr << e.what() << std::endl;
//                        std::cerr << "This is probably a bug in the CoreML Execution Provider." << std::endl;
//                    }
//#endif
//                    if(model == nullptr) {
                        model = OgaModel::Create(model_path.c_str());
//                    }
                    
                    // 5. Create Tokenizer
                    tokenizer = OgaTokenizer::Create(*model);
                                        
                    // 7. Load Templates
                    if(chat_template == "") {
                        chat_template = LoadChatTemplate(model_path);
                    }
                    model_created = get_created_timestamp();
                } catch (const std::exception& e) {
                    std::cerr << "Failed to load model: " << e.what() << std::endl;
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
    int max_position_embeddings;
    RerankingMode ranking_mode_embeddings;
    int cls_id_embeddings = 101;
    int sep_id_embeddings = 102;
    
    if (embedding_model_path.length() != 0) {
        if (fs::exists(embedding_model_path)) {
            if (fs::is_regular_file(embedding_model_path)) {
                // 1.b Initialize Embedding and Session (Load once)
                std::cerr << "[Embedding] Loading from " << embedding_model_path << std::endl;
                embedding_fingerprint = get_system_fingerprint(embedding_model_path, "directml");
                try {
                    embeddings_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Embeddings");
#ifdef WIN32
                    embedding_modelName = get_model_name(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()));
#else
                    embedding_modelName = get_model_name(fs::path(embedding_model_path).parent_path());
#endif
                    Ort::SessionOptions session_options;
                    session_options.SetIntraOpNumThreads(intra_op_threads);
                    
#if defined(__APPLE__)
//                    std::unordered_map<std::string, std::string> provider_options;
//                    provider_options["ModelFormat"] = "MLProgram";
//                    provider_options["MLComputeUnits"] = "ALL";
//                    provider_options["RequireStaticShapes"] = "0";
//                    provider_options["EnableSubgraphs"] = "0";
//                    session_options.AppendExecutionProvider("CoreML", provider_options);
#endif

                    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                    
                    session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");

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
                    }
                    for (size_t i = 0; i < num_output_nodes; i++) {
                        auto output_name_ptr = embeddings_session->GetOutputNameAllocated(i, allocator);
                        output_node_names.push_back(output_name_ptr.get());
                    }
                    for (const auto& name : input_node_names) {
                        input_names_c_array.push_back(name.c_str());
                    }
                    for (const auto& name : output_node_names) {
                        output_names_c_array.push_back(name.c_str());
                    }
#ifdef WIN32
                    embeddings_tokenizer = LoadTokenizer(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()));
                    max_position_embeddings = LoadMaxPositionEmbeddings(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()));
                    ranking_mode_embeddings = LoadRerankingMode(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()));
                    LoadSpecialTokenIds(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()),
                                        ranking_mode_embeddings,
                                        cls_id_embeddings,
                                        sep_id_embeddings);
#else
                    embeddings_tokenizer = LoadTokenizer(fs::path(embedding_model_path).parent_path());
                    max_position_embeddings = LoadMaxPositionEmbeddings(fs::path(embedding_model_path).parent_path());
                    ranking_mode_embeddings = LoadRerankingMode(fs::path(embedding_model_path).parent_path());
                    LoadSpecialTokenIds(fs::path(embedding_model_path).parent_path(),
                                        ranking_mode_embeddings,
                                        cls_id_embeddings,
                                        sep_id_embeddings);
#endif
                    embedding_model_created = get_created_timestamp();
                } catch (const std::exception& e) {
                    std::cerr << "Failed to load model: " << e.what() << std::endl;
                    return 1;
                }
            }
        }
    }
    
    std::string reranking_fingerprint;
    long long reranking_model_created = 0;
    std::string reranking_modelName;
    std::unique_ptr<Ort::Session> rerank_session;
    std::unique_ptr<Ort::Env> rerank_env;
    size_t num_reranking_input_nodes = 0;
    size_t num_reranking_output_nodes = 0;
    std::vector<std::string> reranking_input_node_names;
    std::vector<std::string> reranking_output_node_names;
    Ort::AllocatorWithDefaultOptions rerank_allocator;
    std::vector<int64_t> reranking_input_shape = {1}; // Batch size 1
    std::vector<const char*> reranking_input_names_c_array;
    std::vector<const char*> reranking_output_names_c_array;
    std::unique_ptr<Tokenizer> rerank_tokenizer;
    int rerank_max_position_embeddings;
    RerankingMode ranking_mode;
    int rerank_cls_id = 101;
    int rerank_sep_id = 102;
    
    if (reranker_model_path.length() != 0) {
        if (fs::exists(reranker_model_path)) {
            if (fs::is_regular_file(reranker_model_path)) {
                // 1.b Initialize Reranking and Session (Load once)
                std::cerr << "[Rerank] Loading from " << reranker_model_path << std::endl;
                reranking_fingerprint = get_system_fingerprint(reranker_model_path, "directml");
                try {
                    rerank_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Rerank");
#ifdef WIN32
                    reranking_modelName = get_model_name(wchar_to_utf8(fs::path(reranker_model_path).parent_path().c_str()));
#else
                    reranking_modelName = get_model_name(fs::path(reranker_model_path).parent_path());
#endif
                    Ort::SessionOptions session_options;
                    session_options.SetIntraOpNumThreads(intra_op_threads);
                    
#if defined(__APPLE__)
//                    std::unordered_map<std::string, std::string> provider_options;
//                    provider_options["ModelFormat"] = "MLProgram";
//                    provider_options["MLComputeUnits"] = "ALL";
//                    provider_options["RequireStaticInputShapes"] = "0";
//                    provider_options["EnableOnSubgraphs"] = "0";
//                    session_options.AppendExecutionProvider("CoreML", provider_options);
#endif
                    
                    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                    
                    session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
                    
                    Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*)session_options, OrtGetApiBase()));
#ifdef WIN32
                    rerank_session = std::make_unique<Ort::Session>(*rerank_env, reranker_model_path_u16.c_str(), session_options);
#else
                    rerank_session = std::make_unique<Ort::Session>(*rerank_env, reranker_model_path.c_str(), session_options);
#endif
                    num_reranking_input_nodes = rerank_session->GetInputCount();
                    num_reranking_output_nodes = rerank_session->GetOutputCount();
                    for (size_t i = 0; i < num_reranking_input_nodes; i++) {
                        auto input_name_ptr = rerank_session->GetInputNameAllocated(i, rerank_allocator);
                        reranking_input_node_names.push_back(input_name_ptr.get());
                    }
                    for (size_t i = 0; i < num_reranking_output_nodes; i++) {
                        auto output_name_ptr = rerank_session->GetOutputNameAllocated(i, rerank_allocator);
                        reranking_output_node_names.push_back(output_name_ptr.get());
                    }
                    for (const auto& name : reranking_input_node_names) {
                        reranking_input_names_c_array.push_back(name.c_str());
                    }
                    for (const auto& name : reranking_output_node_names) {
                        reranking_output_names_c_array.push_back(name.c_str());
                    }
#ifdef WIN32
                    rerank_tokenizer = LoadTokenizer(wchar_to_utf8(fs::path(reranker_model_path).parent_path().c_str()));
                    rerank_max_position_embeddings = LoadMaxPositionEmbeddings(wchar_to_utf8(fs::path(reranker_model_path).parent_path().c_str()));
                    ranking_mode = LoadRerankingMode(wchar_to_utf8(fs::path(reranker_model_path).parent_path().c_str()));
                    LoadSpecialTokenIds(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()),
                                        ranking_mode,
                                        rerank_cls_id,
                                        rerank_sep_id);
#else
                    rerank_tokenizer = LoadTokenizer(fs::path(reranker_model_path).parent_path());
                    rerank_max_position_embeddings = LoadMaxPositionEmbeddings(fs::path(reranker_model_path).parent_path());
                    ranking_mode = LoadRerankingMode(fs::path(reranker_model_path).parent_path());
                    LoadSpecialTokenIds(fs::path(embedding_model_path).parent_path(),
                                        ranking_mode,
                                        rerank_cls_id,
                                        rerank_sep_id);
#endif
                    reranking_model_created = get_created_timestamp();
                } catch (const std::exception& e) {
                    std::cerr << "Failed to load model: " << e.what() << std::endl;
                    return 1;
                }
            }
        }
    }
    
    const std::string instruction = "Given a web search query, retrieve relevant passages that answer the query";

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
                double repetition_penalty = 1.2;
                unsigned int n = 1;
                bool is_stream = false;
                std::string guidance_string_type;
                std::string guidance_string;
                
                before_run_inference(req.body,
                                     prompt,
                                     &max_tokens,
                                     &top_k,
                                     &top_p,
                                     &temperature,
                                     &repetition_penalty,
                                     &n,
                                     &is_stream,
                                     tokenizer.get(),
                                     chat_template,
                                     guidance_string_type,
                                     guidance_string);
                
                if(is_stream) {
                    std::string req_id = get_openai_style_id();
                    
                    // 1. Extract raw pointers safely.
                    // Since 'model' and 'tokenizer' live in main() for the lifetime of the app,
                    // these pointers will remain valid while the stream runs.
                    OgaModel* raw_model = model.get();
                    OgaTokenizer* raw_tokenizer = tokenizer.get();
                    
                    // 2. Explicitly capture EVERYTHING by value (copy) or by safe pointer.
                    // We removed the dangerous `&` default capture.
                    res.set_chunked_content_provider("text/event-stream",
                                                     [
                                                         raw_model,
                                                         raw_tokenizer,
                                                         modelName,
                                                         fingerprint,
                                                         model_created,
                                                         req_id,
                                                         prompt,
                                                         max_tokens,
                                                         top_k,
                                                         top_p,
                                                         temperature,
                                                         repetition_penalty,
                                                         n,
                                                         guidance_string_type,
                                                         guidance_string
                                                     ](size_t offset, httplib::DataSink &sink) {

                        // Send initial role packet (optional but good practice)
                        for (int i = 0; i < n; i++) {
                            std::string role_chunk = create_stream_chunk(i, req_id, modelName, fingerprint, "");
                            sink.write(role_chunk.data(), role_chunk.size());
                        }
                        
                        // Define a callback to handle tokens as they are generated
                        // It's safe to use [&] HERE because this lambda is synchronous and
                        // only executes within the lifespan of the parent lambda.
                        auto token_callback = [&](const std::string& token, unsigned int n) {
                                                       
                            std::string chunk = create_stream_chunk(n, req_id, modelName, fingerprint, token);
                            sink.write(chunk.data(), chunk.size());
                            
                            return true; // Return false to stop inference if needed
                        };

                        run_inference_stream(
                                             raw_model,
                                             raw_tokenizer,
                                             modelName,
                                             fingerprint,
                                             model_created,
                                             max_tokens,
                                             top_k,
                                             top_p,
                                             temperature,
                                             repetition_penalty,
                                             n,
                                             prompt,
                                             guidance_string_type,
                                             guidance_string,
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
                                                              repetition_penalty,
                                                              n,
                                                              prompt,
                                                              guidance_string_type,
                                                              guidance_string
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
            if(reranking_model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = reranking_modelName;
                modelCard["object"] = "model";
                modelCard["created"] = reranking_model_created;
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
        
        // Route: /v1/rerank
        svr.Post("/v1/rerank", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/rerank request received." << std::endl;
            
            try {
                
                if(reranking_model_created == 0) {
                    throw std::invalid_argument("[Rerank] Model not loaded.");
                }
                
                std::string query;
                int top_n = -1;
                std::vector<std::string> documents;
                before_run_reranking(req.body, query, &top_n, documents);
                
                std::string response_json;
                    
                std::vector<RerankItem> items;
                
                // For ColBERT late interaction, pooling_mode must be set to POOLING_COLBERT
                // (e.g., via the '-b' flag when starting the server).
                if (pooling_mode == POOLING_COLBERT) {
                    if (rerank_tokenizer != NULL) {
                        response_json = run_colbert_reranking(
                                                              rerank_session.get(), query, documents, rerank_tokenizer.get(),
                                                              rerank_max_position_embeddings, top_n,
                                                              reranking_input_names_c_array, num_reranking_input_nodes,
                                                              reranking_output_names_c_array, num_reranking_output_nodes,
                                                              ranking_mode,
                                                              rerank_cls_id,
                                                              rerank_sep_id
                                                              );
                    }
                } else {
                    /*
                     Standard Cross-Encoder Reranking
                     */
                    if (rerank_tokenizer != NULL) {
                        std::vector<int> q = rerank_tokenizer->Encode(query);
                        for (size_t i = 0; i < documents.size(); ++i) {
                            std::vector<int> ids;
                            std::vector<int> type_ids;
                            
                            std::vector<int> d = rerank_tokenizer->Encode(documents[i]);
                            
                            switch (ranking_mode) {
                                case RERANKING_MODERNBERT:
                                    ids.reserve(q.size() + d.size() + 3);
                                    ids.push_back(rerank_cls_id); // <cls>
                                    for(int x : q) { ids.push_back(x); }
                                    ids.push_back(rerank_sep_id); // <sep>
                                    for(int x : d) { ids.push_back(x); }
                                    ids.push_back(rerank_sep_id); // <sep>
                                    type_ids.resize(ids.size(), 0);
                                    break;
                                case RERANKING_ROBERTA:
                                    ids.reserve(q.size() + d.size() + 4);
                                    ids.push_back(rerank_cls_id); // <s>
                                    ids.insert(ids.end(), q.begin(), q.end());
                                    ids.push_back(rerank_sep_id); // </s>
                                    ids.push_back(rerank_sep_id); // </s>
                                    ids.insert(ids.end(), d.begin(), d.end());
                                    ids.push_back(rerank_sep_id); // </s>
                                    type_ids.resize(ids.size(), 0);
                                    break;
                                case RERANKING_BERT:
                                    ids.reserve(q.size() + d.size() + 3);
                                    type_ids.reserve(ids.capacity());
                                    ids.push_back(rerank_cls_id); // [CLS]
                                    type_ids.push_back(0);
                                    for(int x : q) { ids.push_back(x); type_ids.push_back(0); }
                                    ids.push_back(rerank_sep_id); // [SEP]
                                    type_ids.push_back(0);
                                    for(int x : d) { ids.push_back(x); type_ids.push_back(1); }
                                    ids.push_back(rerank_sep_id); // [SEP]
                                    type_ids.push_back(1);
                                    break;
                                case RERANKING_LLM:
                                default:
                                {
                                    // 1. Build the prompt exactly as the model expects
                                    std::string prompt = "<Instruct>: " + instruction + "\n" +
                                    "<Query>: " + query + "\n" +
                                    "<Document>: " + documents[i];
                                    
                                    // 2. Encode the single formatted string
                                    ids = rerank_tokenizer->Encode(prompt);
                                }
                                    break;
                            }
                            
                            if (ids.size() > rerank_max_position_embeddings) {
                                ids.resize(rerank_max_position_embeddings - 1);
                                int end_token_id = 2;
                                if (ranking_mode == RERANKING_BERT) end_token_id = 102;
                                ids.push_back(end_token_id);
                                if (!type_ids.empty()) {
                                    type_ids.resize(rerank_max_position_embeddings - 1);
                                    int end_type_id = (ranking_mode == RERANKING_BERT) ? 1 : 0;
                                    type_ids.push_back(end_type_id);
                                }
                            }
                            items.emplace_back(RerankItem({ ids, type_ids }));
                        }
                        response_json = run_reranking(
                                                      rerank_session.get(),
                                                      items, rerank_max_position_embeddings, top_n,
                                                      reranking_input_names_c_array,
                                                      num_reranking_input_nodes,
                                                      reranking_output_names_c_array,
                                                      num_reranking_output_nodes, ranking_mode);
                        
                    }
                    
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
        
        // Route: /v1/embeddings
        svr.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/embeddings request received." << std::endl;
            
            try {
                
                if(embedding_model_created == 0) {
                    throw std::invalid_argument("[Embedding] Model not loaded.");
                }
               
                std::vector<std::string> texts;
                before_run_embeddings(req.body, texts);
                
                std::string response_json;

                switch (pooling_mode) {
                    case POOLING_E2E:
                        response_json = run_embeddings_e2e(
                                                           embeddings_session.get(),
                                                           texts,
                                                           input_names_c_array,
                                                           num_input_nodes,
                                                           output_names_c_array,
                                                           num_output_nodes);
                        break;
                        
                    default:
                        response_json = run_embeddings(
                                                       embeddings_session.get(),
                                                       texts,
                                                       max_position_embeddings,
                                                       input_names_c_array,
                                                       num_input_nodes,
                                                       output_names_c_array,
                                                       num_output_nodes,
                                                       embeddings_tokenizer.get(),
                                                       pooling_mode,
                                                       cls_id_embeddings,
                                                       sep_id_embeddings);
                        break;
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
        
        // Route: /v1/contextualizedembeddings
        auto contextualized_embeddings_handler = [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/contextualizedembeddings request received." << std::endl;
            
            try {
                
                if(embedding_model_created == 0) {
                    throw std::invalid_argument("[Embedding] Contextualized Embedding Model not loaded.");
                }
                
                std::vector<std::string> texts;
                before_run_contextualized_embeddings(req.body, texts);
                
                std::string response_json;
                
                switch (pooling_mode) {
                    case POOLING_E2E:
                        response_json = run_embeddings_e2e(
                                                           embeddings_session.get(),
                                                           texts,
                                                           input_names_c_array,
                                                           num_input_nodes,
                                                           output_names_c_array,
                                                           num_output_nodes);
                        break;
                        
                    default:
                        response_json = run_embeddings(
                                                       embeddings_session.get(),
                                                       texts,
                                                       max_position_embeddings,
                                                       input_names_c_array,
                                                       num_input_nodes,
                                                       output_names_c_array,
                                                       num_output_nodes,
                                                       embeddings_tokenizer.get(),
                                                       pooling_mode,
                                                       cls_id_embeddings,
                                                       sep_id_embeddings);
                        break;
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
            
        };

        svr.Post("/v1/contextualizedembeddings", contextualized_embeddings_handler);
        svr.Post("/v1/contextualized/embeddings", contextualized_embeddings_handler);

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
            double repetition_penalty = 1.2;
            unsigned int n = 1;
            bool is_stream = false;
            std::string guidance_string_type;
            std::string guidance_string;
            
            before_run_inference(request_str,
                                 prompt,
                                 &max_tokens,
                                 &top_k,
                                 &top_p,
                                 &temperature,
                                 &repetition_penalty,
                                 &n,
                                 &is_stream,
                                 tokenizer.get(),
                                 chat_template,
                                 guidance_string_type,
                                 guidance_string);
            
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
                                     repetition_penalty,
                                     n,
                                     prompt,
                                     guidance_string_type,
                                     guidance_string
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
    
    OgaShutdown();
    
    return 0;
}
