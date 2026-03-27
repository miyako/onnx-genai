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
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open file: " + path);

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    std::string data(size, '\0');
    ifs.seekg(0, std::ios::beg);
    ifs.read(&data[0], size);

    return data;
}

// ─── TTS State ────────────────────────────────────────────────────────────────
// Each voice is stored as a flat float array of shape [-1, 1, 256].
// To get the style vector for a given token sequence of length L:
//   style = voice_data[L * 256 ... (L+1) * 256]   (shape [1, 256])
// The number of available length steps = voice_data.size() / 256
struct KokoroVoice {
    std::string            name;
    std::vector<float>     data;      // flat float32, size = steps * 256
    size_t                 steps;     // data.size() / 256
};

// ─── Load vocab from config.json ──────────────────────────────────────────────
// config.json has a "vocab" key: { "phoneme_char": token_id, ... }
static std::unordered_map<std::string, int64_t>
LoadKokoroVocab(const std::string& model_dir) {
    std::unordered_map<std::string, int64_t> vocab;

    fs::path config_path = fs::path(model_dir) / "config.json";
    if (!fs::exists(config_path)) {
        std::cerr << "[TTS] config.json not found in " << model_dir << std::endl;
        return vocab;
    }

    std::string json = LoadBytesFromFile(config_path.string());
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(json.c_str(), json.c_str() + json.size(), &root, &errors)) {
        std::cerr << "[TTS] Failed to parse config.json: " << errors << std::endl;
        return vocab;
    }

    // vocab lives under root["vocab"]
    if (!root.isMember("vocab") || !root["vocab"].isObject()) {
        std::cerr << "[TTS] config.json has no 'vocab' object." << std::endl;
        return vocab;
    }

    const Json::Value& v = root["vocab"];
    for (const auto& key : v.getMemberNames()) {
        vocab[key] = v[key].asInt64();
    }
    std::cout << "[TTS] Loaded vocab: " << vocab.size() << " entries." << std::endl;
    return vocab;
}

struct ModelConfig {
    RerankingMode ranking_mode;
    int max_position_embeddings;
    int cls_id, sep_id;
};

enum class ToolCallState {
    TEXT,       // normal streaming — emit tokens immediately
    TAG_OPEN,   // inside a partial <tool_call> prefix match
    BUFFERING,  // confirmed open tag — accumulate JSON body silently
};

struct SequenceState {
    ToolCallState state = ToolCallState::TEXT;
    std::string   body;         // JSON accumulation buffer (BUFFERING only)
    size_t        sent_offset = 0; // replaces prev_text entirely
    bool          finished = false;
};

// ─── espeak-ng init ────────────────────────────────────────────────────────────
static bool InitEspeak(const std::string& model_dir) {

    fs::path data_path = fs::path(model_dir) / "espeak-ng-data";
    std::string data_path_str = data_path.string();
    const char* path = fs::exists(data_path) ? data_path_str.c_str() : nullptr;

    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, path, 0);
    if (result < 0) {
        std::cerr << "[TTS] espeak_Initialize failed: " << result << std::endl;
        return false;
    }
    espeak_SetVoiceByName("en-us");
    espeak_SetParameter(espeakRATE, 175, 0);
    espeak_SetParameter(espeakPITCH, 50, 0);
    espeak_SetParameter(espeakVOLUME, 100, 0);
    std::cout << "[TTS] espeak-ng initialized." << std::endl;
    return true;
}

// ─── Text → IPA phoneme string ────────────────────────────────────────────────
static std::string TextToPhonemes(const std::string& text) {
    std::string result;
    const void* ptr = text.c_str();
    while (ptr && *(const char*)ptr != '\0') {
        const char* ph = espeak_TextToPhonemes(
            &ptr, espeakCHARS_UTF8, espeakPHONEMES_IPA);
        if (ph) result += ph;
    }
    return result;
}

// ─── IPA phoneme string → token IDs ──────────────────────────────────────────
// Tokens are wrapped with pad token 0 at start and end.
// The "inner" token count (without pads) is what indexes the style vector.
static std::vector<int64_t> PhonemesToTokens(
    const std::string& phonemes,
    const std::unordered_map<std::string, int64_t>& vocab)
{
    std::vector<int64_t> inner; // tokens without BOS/EOS pads

    // Walk UTF-8 — IPA chars can be 1-3 bytes
    size_t i = 0;
    while (i < phonemes.size()) {
        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + (size_t)len > phonemes.size()) continue;
            std::string ch = phonemes.substr(i, len);
            auto it = vocab.find(ch);
            if (it != vocab.end()) {
                inner.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) ++i; // skip unknown
    }

    // Wrap with pad token 0, enforce max context 510 inner tokens
    if (inner.size() > 510) inner.resize(510);

    std::vector<int64_t> tokens;
    tokens.reserve(inner.size() + 2);
    tokens.push_back(0); // BOS pad
    tokens.insert(tokens.end(), inner.begin(), inner.end());
    tokens.push_back(0); // EOS pad

    return tokens;
}

// Loads vocab.json: { "phoneme": token_id, ... }
static std::unordered_map<std::string, int64_t>
LoadVocab(const std::string& model_dir) {
    std::unordered_map<std::string, int64_t> vocab;
    fs::path vocab_path = fs::path(model_dir) / "vocab.json";
    if (!fs::exists(vocab_path)) return vocab;

    std::string json = LoadBytesFromFile(vocab_path.string());
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(json.c_str(), json.c_str() + json.size(), &root, &errors))
        return vocab;

    for (const auto& key : root.getMemberNames()) {
        vocab[key] = root[key].asInt64();
    }
    std::cout << "[TTS] Loaded vocab: " << vocab.size() << " entries." << std::endl;
    return vocab;
}

// ─── Load voices from individual .bin files in voices/ subdirectory ───────────
// Each file is raw float32 with shape [-1, 1, 256].
// The filename stem is the voice name (e.g. "af_heart.bin" → "af_heart").
static std::unordered_map<std::string, KokoroVoice>
LoadKokoroVoices(const std::string& model_dir) {
    std::unordered_map<std::string, KokoroVoice> voices;

    fs::path voices_dir = fs::path(model_dir) / "voices";
    if (!fs::exists(voices_dir) || !fs::is_directory(voices_dir)) {
        std::cerr << "[TTS] voices/ directory not found in " << model_dir << std::endl;
        return voices;
    }

    for (const auto& entry : fs::directory_iterator(voices_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".bin") continue;

        std::string voice_name = entry.path().stem().string();
        std::string raw = LoadBytesFromFile(entry.path().string());

        if (raw.size() % sizeof(float) != 0) {
            std::cerr << "[TTS] Voice file " << voice_name
                << " has unexpected size " << raw.size() << std::endl;
            continue;
        }

        KokoroVoice voice;
        voice.name = voice_name;
        size_t count = raw.size() / sizeof(float);
        voice.data.resize(count);
        std::memcpy(voice.data.data(), raw.data(), raw.size());
        voice.steps = count / 256; // each step is 256 floats

        if (voice.steps == 0) {
            std::cerr << "[TTS] Voice " << voice_name << " has no steps." << std::endl;
            continue;
        }

        voices[voice_name] = std::move(voice);
        std::cout << "[TTS] Loaded voice '" << voice_name
            << "' steps=" << voices[voice_name].steps << std::endl;
    }
    std::cout << "[TTS] Loaded " << voices.size() << " voices." << std::endl;
    return voices;
}

// ─── WAV encoder ──────────────────────────────────────────────────────────────
static std::vector<uint8_t> FloatPCMToWav(
    const float* samples, size_t num_samples, int sample_rate)
{
    std::vector<int16_t> pcm(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float c = std::max(-1.0f, std::min(1.0f, samples[i]));
        pcm[i] = static_cast<int16_t>(c * 32767.0f);
    }

    uint32_t data_size = (uint32_t)(num_samples * sizeof(int16_t));
    uint32_t chunk_size = 36 + data_size;

    std::vector<uint8_t> wav;
    wav.reserve(44 + data_size);

    auto w2 = [&](uint16_t v) {
        wav.push_back(v & 0xFF); wav.push_back(v >> 8);
        };
    auto w4 = [&](uint32_t v) {
        wav.push_back(v & 0xFF);
        wav.push_back((v >> 8) & 0xFF);
        wav.push_back((v >> 16) & 0xFF);
        wav.push_back((v >> 24) & 0xFF);
        };
    auto ws = [&](const char* s) {
        wav.push_back(s[0]); wav.push_back(s[1]);
        wav.push_back(s[2]); wav.push_back(s[3]);
        };

    ws("RIFF"); w4(chunk_size); ws("WAVE");
    ws("fmt "); w4(16);
    w2(1);                          // PCM
    w2(1);                          // mono
    w4((uint32_t)sample_rate);
    w4((uint32_t)sample_rate * 2);  // byte rate (1 ch * 2 bytes)
    w2(2);                          // block align
    w2(16);                         // bits per sample
    ws("data"); w4(data_size);

    const uint8_t* p = reinterpret_cast<const uint8_t*>(pcm.data());
    wav.insert(wav.end(), p, p + data_size);
    return wav;
}

// ─── Core TTS inference ───────────────────────────────────────────────────────
static std::vector<uint8_t> run_tts(
    Ort::Session* session,
    const std::string& text,
    const KokoroVoice& voice,
    float                                                  speed,
    const std::unordered_map<std::string, int64_t>& vocab,
    std::vector<const char*>& input_names,
    size_t                                                 num_inputs,
    std::vector<const char*>& output_names,
    size_t                                                 num_outputs)
{
    // 1. Text → IPA phonemes
    std::string phonemes = TextToPhonemes(text);
    if (phonemes.empty()) {
        std::cerr << "[TTS] Phonemization produced empty output for: "
            << text << std::endl;
        return {};
    }
    std::cout << "[TTS] Phonemes: " << phonemes << std::endl;

    // 2. Phonemes → token IDs (with BOS/EOS pads)
    std::vector<int64_t> tokens = PhonemesToTokens(phonemes, vocab);
    if (tokens.size() <= 2) {
        std::cerr << "[TTS] No tokens after vocab mapping." << std::endl;
        return {};
    }

    // 3. Select style vector indexed by inner token count (tokens minus 2 pads)
    // Clamp to available steps in this voice file.
    size_t inner_len = tokens.size() - 2;
    size_t style_idx = std::min(inner_len, voice.steps - 1);
    // style slice: voice.data[style_idx * 256 ... style_idx * 256 + 256]
    // shape fed to model: [1, 256]
    std::vector<float> style_vec(
        voice.data.begin() + style_idx * 256,
        voice.data.begin() + style_idx * 256 + 256);

    // 4. Build tensors
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    int64_t seq_len = (int64_t)tokens.size();
    std::vector<int64_t> token_dims = { 1, seq_len };
    std::vector<int64_t> style_dims = { 1, 256 };
    std::vector<int64_t> speed_dims = { 1 };
    std::vector<float>   speed_vec = { speed };

    std::vector<Ort::Value> inputs;
    // Input 0: tokens  [1, seq_len]  int64
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, tokens.data(), tokens.size(),
        token_dims.data(), token_dims.size()));
    // Input 1: style   [1, 256]      float32
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, style_vec.data(), style_vec.size(),
        style_dims.data(), style_dims.size()));
    // Input 2: speed   [1]           float32
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, speed_vec.data(), speed_vec.size(),
        speed_dims.data(), speed_dims.size()));

    // 5. Run
    std::vector<Ort::Value> outputs;
    try {
        outputs = session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(), inputs.data(), num_inputs,
            output_names.data(), num_outputs);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[TTS] session->Run failed: " << e.what() << std::endl;
        return {};
    }

    if (outputs.empty()) return {};

    // 6. Output waveform → WAV
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t num_samples = 1;
    for (auto d : shape) num_samples *= (size_t)d;

    const float* waveform = outputs[0].GetTensorMutableData<float>();
    return FloatPCMToWav(waveform, num_samples, 24000);
}

static const std::unordered_map<std::string, RerankingMode> kModelTypeMap = {
    {"xlm-roberta", RERANKING_ROBERTA}, {"roberta", RERANKING_ROBERTA}, {"camembert", RERANKING_ROBERTA},
    {"bert", RERANKING_BERT}, {"mpnet", RERANKING_BERT}, {"deberta-v2", RERANKING_BERT}, {"modernbert", RERANKING_MODERNBERT},
    {"qwen3", RERANKING_LLM}, {"qwen2", RERANKING_LLM}, {"mistral", RERANKING_LLM},
    {"llama", RERANKING_LLM}, {"gemma", RERANKING_LLM}, {"gemma2", RERANKING_LLM}, {"phi3", RERANKING_LLM},
};

static void LoadModelConfig(const std::string& model_path,
    int& cls_id,
    int& sep_id,
    int& positionEmbeddings,
    RerankingMode& ranking_mode) {

    // --- 1. Set defaults ---
    positionEmbeddings = 512;
    ranking_mode = RERANKING_ROBERTA;
    cls_id = 0;   // roberta default
    sep_id = 2;

    // --- 2. Resolve config.json path ---
    fs::path config_path(model_path);
    if (fs::is_directory(config_path)) {
        config_path = config_path / "config.json";
    }
    if (!fs::exists(config_path) || config_path.extension() != ".json") {
        return;
    }

    // --- 3. Parse once ---
    std::string json = LoadBytesFromFile(config_path.string());
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(json.c_str(), json.c_str() + json.size(), &root, &errors)) {
        return;
    }
    if (!root.isObject()) {
        return;
    }

    // --- 4. Resolve ranking_mode from model_type ---
    if (root.isMember("model_type") && root["model_type"].isString()) {
        const std::string model_type = root["model_type"].asString();
        auto it = kModelTypeMap.find(model_type);
        if (it != kModelTypeMap.end()) {
            ranking_mode = it->second;
            std::cout << "[Config] model_type: " << model_type << std::endl;
        }
        else {
            std::cout << "[Config] model_type: '" << model_type
                << "' unrecognized, defaulting to roberta" << std::endl;
        }
    }

    // --- 5. Set architecture-specific token ID defaults ---
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

    // --- 6. Override token IDs from config if present ---
    if (root.isMember("cls_token_id") && root["cls_token_id"].isNumeric()) {
        cls_id = root["cls_token_id"].asInt();
    }
    else if (root.isMember("bos_token_id") && root["bos_token_id"].isNumeric()) {
        cls_id = root["bos_token_id"].asInt();
    }

    if (root.isMember("sep_token_id") && root["sep_token_id"].isNumeric()) {
        sep_id = root["sep_token_id"].asInt();
    }
    else if (root.isMember("eos_token_id") && root["eos_token_id"].isNumeric()) {
        sep_id = root["eos_token_id"].asInt();
    }

    // --- 7. max_position_embeddings ---
    if (root.isMember("max_position_embeddings") && root["max_position_embeddings"].isNumeric()) {
        positionEmbeddings = root["max_position_embeddings"].asInt();
    }

    std::cout << "[Config] ranking_mode=" << ranking_mode
        << " cls=" << cls_id
        << " sep=" << sep_id
        << " max_pos=" << positionEmbeddings
        << std::endl;
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
#else  // Windows and Linux
    unsigned int logical_cores = std::thread::hardware_concurrency();
    threads = (logical_cores > 4) ? (int)(logical_cores / 2) : (int)logical_cores;
#endif
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

struct ParsedToolCall {
    std::string name;
    std::string arguments;
};

// Parses the JSON content extracted from between <tool_call>…</tool_call>.
// Uses jsoncpp to maintain compatibility across the ONNX module.
static std::vector<ParsedToolCall> parse_tool_call_json(const std::string& json_str) {
    std::vector<ParsedToolCall> results;
    Json::Value parsed;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());

    try {
        if (reader->parse(json_str.c_str(), json_str.c_str() + json_str.size(), &parsed, &errors)) {
            // Normalise to an array
            if (!parsed.isArray()) {
                Json::Value arr(Json::arrayValue);
                arr.append(parsed);
                parsed = arr;
            }

            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";

            for (const auto& call : parsed) {
                if (!call.isMember("name") || !call.isMember("arguments")) continue;
                ParsedToolCall tc;
                tc.name = call["name"].asString();

                if (call["arguments"].isObject()) {
                    tc.arguments = Json::writeString(writer, call["arguments"]);
                }
                else {
                    tc.arguments = call["arguments"].asString();
                }
                results.push_back(std::move(tc));
            }
        }
    }
    catch (...) {
        // Malformed JSON — return empty, callers treat this as not-a-tool-call
    }
    return results;
}

static // Helper to read the template file from the model directory
std::string LoadChatTemplate(const std::string& model_path) {
    fs::path path(model_path);
    fs::path chat_template_path = path;

    if (fs::is_directory(path)) {
        chat_template_path = path / "chat_template.jinja";
    }

    if (fs::exists(chat_template_path) && chat_template_path.extension() == ".jinja") {
        return LoadBytesFromFile(chat_template_path.string());
    }

    return "";
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
        std::string blob = LoadBytesFromFile(json_path.string());
        return Tokenizer::FromBlobJSON(blob);
    }

    // 3. Fallback to SentencePiece
    if (fs::exists(model_file_path) && model_file_path.extension() == ".model") {
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
    const float* flat_hidden,   // raw ORT output pointer
    const std::vector<int64_t>& attention_mask,
    int                              batch_size,
    int                              max_seq_len,
    int                              hidden_dim
) {
    Eigen::MatrixXf out(batch_size, hidden_dim);

#pragma omp parallel for
    for (long i = 0; i < batch_size; ++i) {

        // Zero-cost view into the correct row-slice of the flat buffer
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            hidden(flat_hidden + i * max_seq_len * hidden_dim, max_seq_len, hidden_dim);

        // Build mask vector for this item
        Eigen::VectorXf mask_f(max_seq_len);
        for (int j = 0; j < max_seq_len; ++j) {
            mask_f(j) = static_cast<float>(attention_mask[i * max_seq_len + j]);
        }

        float count = mask_f.sum();
        if (count > 1e-9f) {
            out.row(i) = mask_f.transpose() * hidden;
            out.row(i) /= count;
        }
        else {
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
    fprintf(stderr, " -%c path     : %s\n", 'm', "model");
    fprintf(stderr, " -%c path     : %s\n", 'e', "embedding model");
    fprintf(stderr, " -%c path     : %s\n", 'r', "reranker model");
    fprintf(stderr, " -%c path     : %s\n", 'T', "text to speach model");
    fprintf(stderr, " -%c path     : %s\n", 't', "chat template");
    fprintf(stderr, " -%c          : %s\n", 'j', "chat template from stdin");
    fprintf(stderr, " %c           : %s\n", 'd', "pooling=e2e");
    fprintf(stderr, " %c           : %s\n", 'b', "pooling=multi-vector");
    fprintf(stderr, " %c           : %s\n", 'l', "pooling=last-token");
    fprintf(stderr, " %c           : %s\n", 'c', "pooling=cls");
    fprintf(stderr, " %c           : %s\n", 's', "server");
    fprintf(stderr, " %c           : %s\n", 'p', "server listening port (default=8080)");
    fprintf(stderr, " %c           : %s\n", 'h', "server host (default=127.0.0.1)  ");
    fprintf(stderr, " -%c path     : %s\n", 'i', "input");
    fprintf(stderr, " %c           : %s\n", '-', "use stdin for input");
    fprintf(stderr, " -%c path     : %s\n", 'o', "output (default=stdout)");
}

extern OPTARG_T optarg;
extern int optind, opterr, optopt;

#ifdef WIN32
OPTARG_T optarg = 0;
int opterr = 1;
int optind = 1;
int optopt = 0;
int getopt(int argc, OPTARG_T* argv, OPTARG_T opts) {

    static int sp = 1;
    register int c;
    register OPTARG_T cp;

    if (sp == 1)
        if (optind >= argc ||
            argv[optind][0] != '-' || argv[optind][1] == '\0')
            return(EOF);
        else if (wcscmp(argv[optind], L"--") == NULL) {
            optind++;
            return(EOF);
        }
    optopt = c = argv[optind][sp];
    if (c == ':' || (cp = wcschr(opts, c)) == NULL) {
        ERR(L": illegal option -- ", c);
        if (argv[optind][++sp] == '\0') {
            optind++;
            sp = 1;
        }
        return('?');
    }
    if (*++cp == ':') {
        if (argv[optind][sp + 1] != '\0')
            optarg = &argv[optind++][sp + 1];
        else if (++optind >= argc) {
            ERR(L": option requires an argument -- ", c);
            sp = 1;
            return('?');
        }
        else
            optarg = argv[optind++];
        sp = 1;
    }
    else {
        if (argv[optind][++sp] == '\0') {
            sp = 1;
            optind++;
        }
        optarg = NULL;
    }
    return(c);
}
#define ARGS (OPTARG_T)L"m:e:r:i:o:T:sp:jt:bcld-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:e:r:i:o:T:sp:jt:bcld-h"
#define _atoi atoi
#define _atof atof
#endif

#pragma mark -

static long long get_created_timestamp() {
    // std::time(nullptr) returns the current time as a time_t (seconds since epoch)
    return static_cast<long long>(std::time(nullptr));
}

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

static void parse_request_reranking(const std::string& json,
    std::string& query,
    int& top_n,
    std::vector<std::string>& documents
) {

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
        json.c_str() + json.size(),
        &root,
        &errors);

    if (parse)
    {
        if (root.isObject())
        {
            Json::Value query_node = root["query"];
            if (query_node.isString())
            {
                query = query_node.asString();
            }
            Json::Value top_n_node = root["top_n"];
            if (top_n_node.isNumeric())
            {
                top_n = top_n_node.asInt();
            }

            Json::Value documents_node = root["documents"];
            if (documents_node.isArray())
            {
                for (Json::Value::const_iterator it = documents_node.begin(); it != documents_node.end(); it++)
                {
                    if (it->isString())
                    {
                        std::string document = it->asString();
                        documents.push_back(document);
                    }
                }
            }
        }
    }
}

static void parse_request_contextualized_embeddings(const std::string& json,
    std::vector<std::string>& inputs) {

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
        json.c_str() + json.size(),
        &root,
        &errors);

    if (parse && root.isObject())
    {
        // Voyage AI uses "inputs" (plural)
        Json::Value inputs_node = root["inputs"];

        // fallback for 4D AI Kit which uses "inout" (singular)
        inputs_node = inputs_node.isArray() ? inputs_node : root["input"];

        if (inputs_node.isArray())
        {
            // Iterate over documents (each document is an array of chunks)
            for (Json::Value::const_iterator it_doc = inputs_node.begin(); it_doc != inputs_node.end(); ++it_doc)
            {
                const Json::Value& chunk_array = *it_doc;
                if (chunk_array.isArray())
                {
                    // 1. Reconstruct the full document by concatenating its chunks
                    std::string full_document;
                    for (Json::Value::const_iterator it_chunk = chunk_array.begin(); it_chunk != chunk_array.end(); ++it_chunk)
                    {
                        if (it_chunk->isString()) {
                            full_document += it_chunk->asString();
                        }
                    }

                    // 2. Flatten the request: create a contextualized input for each chunk
                    for (Json::Value::const_iterator it_chunk = chunk_array.begin(); it_chunk != chunk_array.end(); ++it_chunk)
                    {
                        if (it_chunk->isString()) {
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

static void parse_request_embeddings(const std::string& json,
    std::vector<std::string>& inputs) {

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parse = reader->parse(json.c_str(),
        json.c_str() + json.size(),
        &root,
        &errors);

    if (parse)
    {
        if (root.isObject())
        {
            Json::Value input_node = root["input"];
            if (input_node.isString())
            {
                inputs.push_back(input_node.asString());
            }
            if (input_node.isArray())
            {
                for (Json::ValueIterator i = input_node.begin(); i != input_node.end(); ++i)
                {
                    const auto& node = *i;
                    if (node.isString())
                    {
                        inputs.push_back(node.asString());
                    }
                }
            }
        }
    }
}

static void parse_request(
    const std::string& json,
    std::string& prompt,
    unsigned int& max_tokens,
    unsigned int& top_k,
    double& top_p,
    double& temperature,
    double& repetition_penalty,
    unsigned int& n,
    bool& is_stream,
    bool& has_tools,
    std::string& tools_str,
    OgaTokenizer& tokenizer,
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

    if (parse)
    {
        if (root.isObject())
        {
            // --- Tool Handling ---
            if (root.isMember("tools") && root["tools"].isArray() && !root["tools"].empty()) {
                has_tools = true;
                Json::StreamWriterBuilder w;
                w["indentation"] = "";
                tools_str = Json::writeString(w, root["tools"]);
            }

            Json::Value messages_node = root["messages"];
            if (messages_node.isArray())
            {
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "";
                std::string messages_json = Json::writeString(writer, messages_node);

                const char* tools_ptr = has_tools ? tools_str.c_str() : nullptr;
                prompt = tokenizer.ApplyChatTemplate(chat_template.c_str(), messages_json.c_str(), tools_ptr, true);
            }
            Json::Value top_p_node = root["top_p"];
            if (top_p_node.isNumeric())
            {
                top_p = top_p_node.asDouble();
            }
            Json::Value top_k_node = root["top_k"];
            if (top_k_node.isNumeric())
            {
                top_k = top_k_node.asInt();
            }
            Json::Value max_tokens_node = root["max_tokens"];
            if (max_tokens_node.isNumeric())
            {
                max_tokens = max_tokens_node.asInt();
            }
            Json::Value repetition_penalty_node = root["repetition_penalty"];
            if (repetition_penalty_node.isNumeric())
            {
                repetition_penalty = repetition_penalty_node.asDouble();
            }
            /*
             only these are set by AI-Kit
             */
            Json::Value temperature_node = root["temperature"];
            if (temperature_node.isNumeric())
            {
                temperature = temperature_node.asDouble();
            }
            Json::Value n_node = root["n"];
            if (n_node.isNumeric())
            {
                n = n_node.asInt();
            }
            max_tokens_node = root["max_completion_tokens"];
            if (max_tokens_node.isNumeric())
            {
                max_tokens = max_tokens_node.asInt();
            }
            Json::Value stream_node = root["stream"];
            if (stream_node.isBool())
            {
                is_stream = stream_node.asBool();
            }
            Json::Value response_format_node = root["response_format"];
            if (response_format_node.isObject())
            {
                Json::Value response_format_type_node = response_format_node["type"];
                if (response_format_type_node.isString())
                {
                    std::string response_format_type = response_format_type_node.asString();
                    if (response_format_type == "json_schema") {
                        Json::Value json_schema_node = response_format_node["json_schema"];
                        if (json_schema_node.isObject())
                        {
                            Json::Value schema_node = json_schema_node["schema"];
                            if (schema_node.isObject())
                            {
                                Json::StreamWriterBuilder writer;
                                writer["indentation"] = "";
                                guidance_string = Json::writeString(writer, schema_node);
                                guidance_string_type = "json_schema";
                            }
                        }
                    }
                    if (response_format_type == "regex") {
                        Json::Value regex_node = response_format_node["regex"];
                        if (regex_node.isString())
                        {
                            guidance_string = regex_node.asString();
                            guidance_string_type = "regex";
                        }
                    }
                    if (response_format_type == "lark_grammar") {
                        Json::Value lark_grammar_node = response_format_node["lark_grammar"];
                        if (lark_grammar_node.isString())
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
    std::string& query,
    int& top_n,
    std::vector<std::string>& documents
) {
    parse_request_reranking(request_body, query, top_n, documents);
}

static void before_run_contextualized_embeddings(
    const std::string& request_body,
    std::vector<std::string>& inputs
) {
    parse_request_contextualized_embeddings(request_body, inputs);
}

static void before_run_embeddings(
    const std::string& request_body,
    std::vector<std::string>& inputs
) {
    parse_request_embeddings(request_body, inputs);
}

static void before_run_inference(
    const std::string& request_body,
    std::string& prompt,
    unsigned int& max_tokens,
    unsigned int& top_k,
    double& top_p,
    double& temperature,
    double& repetition_penalty,
    unsigned int& n,
    bool& is_stream,
    bool& has_tools,
    std::string& tools_str,
    OgaTokenizer& tokenizer,
    std::string& chat_template,
    std::string& guidance_string_type,
    std::string& guidance_string) {

    parse_request(request_body, prompt, max_tokens, top_k, top_p, temperature, repetition_penalty, n, is_stream, has_tools, tools_str, tokenizer, chat_template, guidance_string_type, guidance_string);
}

static std::unordered_set<int32_t> BuildStopTokenSet(OgaTokenizer* tokenizer) {
    // ToTokenId returns this sentinel for unknown tokens.
    // Typically -1 or 0 depending on the OGA version/model.
    // We probe a string that cannot exist in any real vocabulary.
    const int32_t UNKNOWN_SENTINEL = tokenizer->ToTokenId("<|__oga_unknown_probe__|>");

    static const char* kCandidates[] = {
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|start_header_id|>",
        "<pad>",
        "<bos>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<|end|>"
    };

    std::unordered_set<int32_t> stop_tokens;
    for (const char* token_str : kCandidates) {
        int32_t id = tokenizer->ToTokenId(token_str);
        if (id != UNKNOWN_SENTINEL) {
            stop_tokens.insert(id);
            std::cout << "[StopTokens] '" << token_str << "' -> " << id << std::endl;
        }
    }
    return stop_tokens;
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
    std::string guidance_string,
    bool has_tools,
    const std::unordered_set<int32_t>& stop_tokens
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

    if (guidance_string_type != "") {
        params->SetGuidance(guidance_string_type.c_str(), guidance_string.c_str());
    }

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
        if (generator->IsDone()) break;
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

        std::string finish_reason_local = finish_reason;
        std::string response_text = generated_responses[i];

        // --- TOOL CALL INTERCEPTION ---
        std::vector<ParsedToolCall> tool_calls_parsed;
        if (has_tools) {
            size_t start_tag = response_text.find("<tool_call>");
            size_t end_tag = response_text.find("</tool_call>");

            if (start_tag != std::string::npos && end_tag != std::string::npos) {
                std::string json_str = response_text.substr(start_tag + 11, end_tag - (start_tag + 11));
                tool_calls_parsed = parse_tool_call_json(json_str);
            }
        }

        if (!tool_calls_parsed.empty()) {
            messageNode["content"] = Json::nullValue;

            Json::Value tool_calls_node(Json::arrayValue);
            for (int tc_idx = 0; tc_idx < (int)tool_calls_parsed.size(); ++tc_idx) {
                Json::Value tc(Json::objectValue);
                tc["id"] = "call_" + get_openai_style_id();
                tc["type"] = "function";
                tc["index"] = tc_idx;
                Json::Value func(Json::objectValue);
                func["name"] = tool_calls_parsed[tc_idx].name;
                func["arguments"] = tool_calls_parsed[tc_idx].arguments;
                tc["function"] = func;
                tool_calls_node.append(tc);
            }

            messageNode["tool_calls"] = tool_calls_node;
            finish_reason_local = "tool_calls";
        }
        else {
            messageNode["content"] = response_text.c_str();
        }

        messageNode["refusal"] = Json::nullValue;
        choiceNode["message"] = messageNode;
        choiceNode["logprobs"] = Json::nullValue;
        choiceNode["finish_reason"] = finish_reason_local;
        choicesNode.append(choiceNode);
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
    bool finish,
    Json::UInt64 created) {
    Json::Value root;
    root["id"] = id;
    root["object"] = "chat.completion.chunk";
    root["created"] = created;
    root["model"] = model;
    root["system_fingerprint"] = fingerprint;//Deprecated

    Json::Value choice;
    choice["index"] = n;

    Json::Value delta;
    if (content.empty() && !finish) {
        delta["role"] = "assistant";
    }
    else {
        delta["content"] = content;
    }
    delta["logprobs"] = Json::nullValue;
    choice["delta"] = delta;

    if (finish) {
        choice["finish_reason"] = "stop";
    }
    else {
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
    bool has_tools,
    const std::unordered_set<int32_t>& stop_tokens,
    std::function<bool(const std::string&, int, bool)> on_token_generated
) {

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

    if (guidance_string_type != "") {
        params->SetGuidance(guidance_string_type.c_str(), guidance_string.c_str());
    }

    // Create Generator
    // Generator is stateful; we need 1 per request.
    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*input_sequences);
    // Create a vector of streams
    // Decoding is stateful; we need 1 decoder per sequence.
//    std::vector<std::string> generated_responses(n, "");
//    std::vector<std::string> previous_text(n, "");
//    std::vector<bool> tool_mode(n, false);
//    std::vector<bool> finished(n, false);
    std::vector<std::string>   generated_responses(n, "");
    std::vector<SequenceState> seq_state(n);

    std::vector<std::unique_ptr<OgaTokenizerStream>> streams;
    for (int i = 0; i < n; i++) {
        streams.push_back(OgaTokenizerStream::Create(*tokenizer));
    }

    // Start Generating
    while (1) {
        generator->GenerateNextToken();
        if (generator->IsDone()) break;
        // Iterate through each sequence (0 to n-1) to collect results
        for (int i = 0; i < n; i++) {
            const auto* seq_data = generator->GetSequenceData(i);
            size_t      seq_len = generator->GetSequenceCount(i);
            if (seq_len == 0) continue;

            int32_t     new_token = seq_data[seq_len - 1];
            bool        hit_stop = false;
            if (stop_tokens.count(new_token)) hit_stop = true;
            const char* token_str = streams[i]->Decode(new_token);
            if (!token_str) continue;

            const std::string tok(token_str);
            generated_responses[i] += tok;
            SequenceState& ss = seq_state[i];

            // ── Replacement for the old tool_mode / TEXT / BUFFERING logic ──

            if (ss.state == ToolCallState::TEXT) {

                if (!has_tools) {
                    // Fast path: just send the new token directly, no substr at all
                    if (!tok.empty() && tok.find("\xef\xbf\xbd") == std::string::npos) {
                        ss.sent_offset = generated_responses[i].size();
                        if (!on_token_generated(tok, i, false)) break;
                    }
                    continue;
                }

                // Unsent tail — a string_view, zero allocation
                const std::string& full = generated_responses[i];
                std::string_view new_text(full.data() + ss.sent_offset,
                    full.size() - ss.sent_offset);

                size_t tag_pos = new_text.find("<tool_call>");
                if (tag_pos == std::string::npos) {
                    // Check for partial tag match at the tail
                    const std::string_view open_tag = "<tool_call>";
                    size_t flush_up_to = new_text.size();
                    for (size_t plen = std::min(new_text.size(), open_tag.size() - 1);
                        plen > 0; --plen) {
                        if (new_text.substr(new_text.size() - plen) == open_tag.substr(0, plen)) {
                            flush_up_to = new_text.size() - plen;
                            break;
                        }
                    }
                    std::string_view safe = new_text.substr(0, flush_up_to);
                    if (!safe.empty() && safe.find("\xef\xbf\xbd") == std::string::npos) {
                        ss.sent_offset += safe.size();  // advance cursor, no copy
                        if (!on_token_generated(std::string(safe), i, false)) break;
                    }
                }
                else {
                    // Stream text before the tag
                    std::string_view before_tag = new_text.substr(0, tag_pos);
                    if (!before_tag.empty() &&
                        before_tag.find("\xef\xbf\xbd") == std::string::npos) {
                        on_token_generated(std::string(before_tag), i, false);
                    }
                    // Freeze cursor at the tag position — no string copy
                    ss.sent_offset = full.size();
                    ss.state = ToolCallState::BUFFERING;
                    std::string_view after_tag = new_text.substr(tag_pos + 11);
                    ss.body += after_tag;
                }
            }
            else if (ss.state == ToolCallState::BUFFERING) {
                ss.body += tok;
                size_t search_from = ss.body.size() > 12 ? ss.body.size() - 12 : 0;
                size_t close_pos = ss.body.find("</tool_call>", search_from);
                if (close_pos != std::string::npos || hit_stop) {
                    std::string json_str = (close_pos != std::string::npos)
                        ? ss.body.substr(0, close_pos)
                        : ss.body;
                    // trim whitespace
                    auto ltrim = json_str.find_first_not_of(" \t\r\n");
                    auto rtrim = json_str.find_last_not_of(" \t\r\n");
                    if (ltrim != std::string::npos)
                        json_str = json_str.substr(ltrim, rtrim - ltrim + 1);
                    if (!json_str.empty()) {
                        ss.finished = true;
                        on_token_generated(json_str, i, true);
                    }
                    ss.state = ToolCallState::TEXT;
                    ss.body.clear();
                }
            }

        }
    }

    // Flush remaining tool calls if abruptly ended (e.g., max length reached)
    for (int i = 0; i < n; i++) {
        SequenceState& ss = seq_state[i];
        if (ss.state == ToolCallState::BUFFERING && !ss.finished && !ss.body.empty()) {
            // Model hit max_length mid-tool-call — flush whatever we have
            on_token_generated(ss.body, i, true);
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

        int last_token_index = -1;
        for (int i = 0; i < max_seq_len; ++i) {
            if (attention_mask[b * max_seq_len + i] == 1) {
                last_token_index = i;   // keep updating — never break early
            }
        }

        if (last_token_index == -1) {
            // Entire mask is zero — degenerate input, emit a zero vector
            batch_embeddings.push_back(std::vector<float>(hidden_size, 0.0f));
            continue;
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
    std::string result;
    result.reserve(64 + batch_size * max_seq_len * 64); // heuristic per token-vector
    result += "{\"object\":\"list\",\"data\":[";

    if (!outputs.empty()) {
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() > 2) {
            const int64_t hidden_size = shape[2];
            const float* floatarr = outputs[0].GetTensorMutableData<float>();
            char num_buf[32];
            bool first_batch = true;

            for (int b = 0; b < batch_size; ++b) {
                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    raw_matrix(floatarr + (b * max_seq_len * hidden_size), max_seq_len, hidden_size);

                Eigen::MatrixXf normalized_matrix = raw_matrix.rowwise().normalized();

                if (!first_batch) result += ',';
                first_batch = false;

                result += "{\"object\":\"embedding\",\"index\":";
                result += std::to_string(b);
                result += ",\"embedding\":[";

                bool first_token = true;
                for (int i = 0; i < max_seq_len; ++i) {
                    if (attention_mask[b * max_seq_len + i] == 0) continue; // skip padding

                    if (!first_token) result += ',';
                    first_token = false;

                    result += '[';
                    for (int j = 0; j < (int)hidden_size; ++j) {
                        if (j > 0) result += ',';
                        snprintf(num_buf, sizeof(num_buf), "%.9g", normalized_matrix(i, j));
                        result += num_buf;
                    }
                    result += ']';
                }

                result += "]}";
            }
        }
    }

    result += "]}";
    return result;
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

    int64_t hidden_dim = shape[2];
    const float* floatarr = outputs[0].GetTensorMutableData<float>();

    // Pass the raw pointer — no matrix copies at all
    Eigen::MatrixXf pooled = mean_pool_batch(
        floatarr, attention_mask, batch_size, max_seq_len, (int)hidden_dim);

    for (int b = 0; b < batch_size; ++b) {
        Eigen::VectorXf final_embedding = l2_normalize(pooled.row(b));
        batch_embeddings.push_back(
            std::vector<float>(final_embedding.data(),
                final_embedding.data() + final_embedding.size()));
    }
    return batch_embeddings;
}

static std::string run_reranking(
    Ort::Session* session,
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
        Ort::RunOptions{ nullptr },
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
    }
    else {
        final_scores = (1.0f + (-logits_mat.col(0)).array().exp()).inverse();
    }

    // 7. Sort & Build JSON
    std::vector<RerankResult> results;
    results.reserve(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        results.push_back({ b, final_scores[b] });
    }

    auto sorter = [](const RerankResult& a, const RerankResult& b) {
        return a.score > b.score;
        };

    if (top_n > 0 && top_n < batch_size) {
        std::partial_sort(results.begin(), results.begin() + top_n, results.end(), sorter);
        results.resize(top_n);
    }
    else {
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
}

static std::string run_embeddings(
    Ort::Session* session,
    std::vector<std::string>& inputs,
    int max_position_embeddings,
    std::vector<const char*>& input_names_c_array,
    size_t num_input_nodes,
    std::vector<const char*>& output_names_c_array,
    size_t num_output_nodes,
    Tokenizer* tokenizer,
    PoolingMode pooling_mode,
    int cls_id,
    int sep_id,
    bool using_coreml = false)
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

        if (using_coreml) {
            max_seq_len = max_position_embeddings;  // force exact shape
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
            Ort::RunOptions{ nullptr },
            input_names_c_array.data(),
            input_tensors.data(),
            num_input_nodes,
            output_names_c_array.data(),
            num_output_nodes
        );

        // 6. Pooling & Build JSON
        if (pooling_mode == POOLING_COLBERT) {
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

        // Pre-size the result string to avoid repeated reallocations.
        // Heuristic: each float ~10 chars + punctuation overhead.
        std::string result;
        const size_t embedding_dim = batch_embeddings.empty() ? 0 : batch_embeddings[0].size();
        result.reserve(64 + batch_size * (32 + embedding_dim * 11));

        result += "{\"object\":\"list\",\"data\":[";

        char num_buf[32];
        for (int b = 0; b < batch_size; ++b) {
            if (b > 0) result += ',';
            result += "{\"object\":\"embedding\",\"index\":";
            result += std::to_string(b);
            result += ",\"embedding\":[";

            const auto& emb = batch_embeddings[b];
            for (size_t i = 0; i < emb.size(); ++i) {
                if (i > 0) result += ',';
                // snprintf is locale-independent and always uses '.' as decimal separator
                snprintf(num_buf, sizeof(num_buf), "%.9g", emb[i]);
                result += num_buf;
            }

            result += "]}";
        }

        result += "]}";
        return result;

    }
    catch (const std::exception& e) {
        throw; // Controller handles the JSON error formatting
    }
}

static std::string run_embeddings_e2e(
    Ort::Session* session,
    std::vector<std::string>& inputs,
    std::vector<const char*>& input_names_c_array,
    size_t                     num_input_nodes,
    std::vector<const char*>& output_names_c_array,
    size_t                     num_output_nodes)
{
    if (inputs.empty()) {
        return "{\"object\":\"list\",\"data\":[]}";
    }

    const OrtApi& api = Ort::GetApi();

    // --- 1. Build a single [N] string tensor holding all inputs ---
    OrtAllocator* allocator = nullptr;
    OrtStatus* status = api.GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) {
        api.ReleaseStatus(status);
        return "{\"object\":\"list\",\"data\":[]}";
    }

    const int64_t batch_size = static_cast<int64_t>(inputs.size());
    int64_t input_shape[] = { batch_size };

    // Build a C-string pointer array that FillStringTensorElement expects
    std::vector<const char*> input_cstrs;
    input_cstrs.reserve(batch_size);
    for (const auto& s : inputs) {
        input_cstrs.push_back(s.c_str());
    }

    OrtValue* raw_tensor_ptr = nullptr;
    status = api.CreateTensorAsOrtValue(
        allocator,
        input_shape,
        1,                                       // rank = 1 (flat batch)
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
        &raw_tensor_ptr
    );
    if (status != nullptr) {
        std::cerr << "[E2E] CreateTensorAsOrtValue failed: "
            << api.GetErrorMessage(status) << std::endl;
        api.ReleaseStatus(status);
        return "{\"object\":\"list\",\"data\":[]}";
    }

    // Fill all strings in one call
    status = api.FillStringTensor(raw_tensor_ptr, input_cstrs.data(), batch_size);
    if (status != nullptr) {
        std::cerr << "[E2E] FillStringTensor failed: "
            << api.GetErrorMessage(status) << std::endl;
        api.ReleaseStatus(status);
        api.ReleaseValue(raw_tensor_ptr);
        return "{\"object\":\"list\",\"data\":[]}";
    }

    // --- 2. Single Run() for the whole batch ---
    Ort::Value input_tensor(raw_tensor_ptr);
    std::vector<Ort::Value> outputs;
    try {
        outputs = session->Run(
            Ort::RunOptions{ nullptr },
            input_names_c_array.data(),
            &input_tensor,
            num_input_nodes,
            output_names_c_array.data(),
            num_output_nodes
        );
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[E2E] session->Run failed: " << e.what() << std::endl;
        return "{\"object\":\"list\",\"data\":[]}";
    }

    if (outputs.empty()) {
        return "{\"object\":\"list\",\"data\":[]}";
    }

    // --- 3. Slice output [N, dim] and build JSON ---
    // Output shape is [batch_size, embedding_dim]
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 2) {
        std::cerr << "[E2E] Unexpected output rank: " << shape.size() << std::endl;
        return "{\"object\":\"list\",\"data\":[]}";
    }

    const int64_t embed_dim = shape[1];
    const float* data = outputs[0].GetTensorMutableData<float>();

    // Pre-allocate: each item is ~32 chars of envelope + embed_dim * ~11 chars per float
    std::string result;
    result.reserve(64 + static_cast<size_t>(batch_size) *
        (32 + static_cast<size_t>(embed_dim) * 11));
    result += "{\"object\":\"list\",\"data\":[";

    char num_buf[32];
    for (int64_t b = 0; b < batch_size; ++b) {
        if (b > 0) result += ',';
        result += "{\"object\":\"embedding\",\"index\":";
        result += std::to_string(b);
        result += ",\"embedding\":[";

        const float* row = data + b * embed_dim;   // pointer arithmetic into [N, dim]
        for (int64_t i = 0; i < embed_dim; ++i) {
            if (i > 0) result += ',';
            snprintf(num_buf, sizeof(num_buf), "%.9g", row[i]);
            result += num_buf;
        }
        result += "]}";
    }

    result += "]}";
    return result;
}

static std::string run_colbert_reranking(
    Ort::Session* session,
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
            ids.push_back(sep_id);
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
        Ort::RunOptions{ nullptr },
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
    for (int i = 0; i < max_seq_len; ++i) {
        if (flat_attention_mask[i] == 1) q_len++;
    }

    Eigen::MatrixXf Q_valid(q_len, hidden_size);
    if (q_len > 0) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            raw_Q(data, max_seq_len, hidden_size);
        int q_idx = 0;
        for (int i = 0; i < max_seq_len; ++i) {
            if (flat_attention_mask[i] == 1) {
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
        for (int i = 0; i < max_seq_len; ++i) {
            if (flat_attention_mask[b * max_seq_len + i] == 1) d_len++;
        }

        if (q_len == 0 || d_len == 0) {
            results.push_back({ b - 1, 0.0f });
            continue;
        }

        Eigen::MatrixXf D_valid(d_len, hidden_size);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            raw_D(data + b * max_seq_len * hidden_size, max_seq_len, hidden_size);

        int d_idx = 0;
        for (int i = 0; i < max_seq_len; ++i) {
            if (flat_attention_mask[b * max_seq_len + i] == 1) {
                D_valid.row(d_idx++) = raw_D.row(i).normalized();
            }
        }

        // MaxSim math: Query tokens (rows) x Doc tokens (cols) -> Resulting in [q_len, d_len]
        Eigen::MatrixXf Sim = Q_valid * D_valid.transpose();

        // For each query token, find max similarity across doc tokens, then sum for total score
        float score = Sim.rowwise().maxCoeff().sum();
        results.push_back({ b - 1, score });
    }

    // 8. Sort and Build JSON
    auto sorter = [](const RerankResult& a, const RerankResult& b) {
        return a.score > b.score;
        };

    if (top_n > 0 && top_n < (int)results.size()) {
        std::partial_sort(results.begin(), results.begin() + top_n, results.end(), sorter);
        results.resize(top_n);
    }
    else {
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
}

static std::string MakeErrorJson(const std::string& message, const std::string& type = "invalid_request_error") {
    Json::Value root(Json::objectValue);
    Json::Value err(Json::objectValue);
    err["message"] = message;
    err["type"] = type;
    err["param"] = Json::nullValue;
    err["code"] = Json::nullValue;
    root["error"] = err;
    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    return Json::writeString(w, root);
}

std::string Phonemize(const std::string& text) {
    std::string result;
    // espeak-ng writes phonemes via callback or returns them
    // depending on which API you use — TextToPhonemes is simplest
    const char* phonemes = espeak_TextToPhonemes(
        (const void**)&text, espeakCHARS_UTF8, espeakPHONEMES_IPA);
    if (phonemes) result = phonemes;
    return result;
}

#pragma mark -

int main(int argc, OPTARG_T argv[]) {

#ifdef WIN32
    std::wstring model_path_u16;
    std::wstring embedding_model_path_u16;
    std::wstring reranker_model_path_u16;
    std::wstring tts_model_path_u16;
#endif
    std::string model_path;           // -m
    std::string embedding_model_path; // -e
    std::string reranker_model_path;  // -r
    std::string chat_template;        // -j
    std::string tts_model_path;       // -T
    OPTARG_T input_path = NULL;      // -i
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
        switch (ch) {
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
        case 'T':
#ifdef WIN32
            tts_model_path_u16 = optarg;
            tts_model_path = wchar_to_utf8(tts_model_path_u16.c_str());
#else
            tts_model_path = optarg;
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
            if (ch == 'j') {
                chat_template = std::string((const char*)cli_request_json.data(), cli_request_json.size());
            }
        }
        break;
        case 't':
        {
            chat_template_path = optarg;
            if (chat_template_path != NULL) {
                FILE* f = _fopen(chat_template_path, _rb);
                if (f) {
                    std::vector<unsigned char> chat_template_string(0);
                    fseek(f, 0, SEEK_END);
                    size_t len = (size_t)ftell(f);
                    fseek(f, 0, SEEK_SET);
                    chat_template_string.resize(len);
                    fread(chat_template_string.data(), 1, chat_template_string.size(), f);
                    fclose(f);
                    chat_template = std::string((const char*)chat_template_string.data(), chat_template_string.size());
                }
            }
        }
        break;
        default:
            usage();
            return 1;  // main returns, stack unwinds, all destructors run
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

    std::unordered_set<int32_t> stop_tokens;
    bool embedding_coreml = false;

#define USE_COREML_FOR_EMBEDDINGS 0
    /*
     INT8 through CoreML is generally not worth the overhead.
     The right candidates for CoreML acceleration are larger unquantized models where the ANE's fp16 throughput beats CPU fp32, and where the graph is clean enough that CoreML can take the majority of nodes without excessive partition boundaries.
     */

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

                    if (tokenizer) {
                        stop_tokens = BuildStopTokenSet(tokenizer.get());
                    }

                    // 7. Load Templates
                    if (chat_template == "") {
                        chat_template = LoadChatTemplate(model_path);
                    }
                    model_created = get_created_timestamp();
                }
                catch (const std::exception& e) {
                    std::cerr << "Failed to load model: " << e.what() << std::endl;
                }
            }
        }
    }

    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

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
    std::vector<int64_t> input_shape = { 1 }; // Batch size 1
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
                    // ── 1. Throwaway session — CPU only, just to read dim names ──────────────
                    std::vector<std::string> sym_dim_names;
                    {
                        Ort::SessionOptions probe_opts;
                        probe_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
                        Ort::Session probe(*embeddings_env, embedding_model_path.c_str(), probe_opts);

                        const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
                        size_t input_count = probe.GetInputCount();
                        OrtStatus* s = nullptr;

                        for (size_t i = 0; i < input_count; i++) {
                            // Get OrtTypeInfo* — must stay alive until we finish reading strings
                            OrtTypeInfo* type_info_ptr = nullptr;
                            s = ort_api->SessionGetInputTypeInfo(probe, i, &type_info_ptr);
                            if (!type_info_ptr) continue;
                            // Cast to tensor shape info — this is a non-owning view into type_info_ptr
                            const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
                            s = ort_api->CastTypeInfoToTensorInfo(type_info_ptr, &tensor_info);
                            if (tensor_info) {
                                size_t dim_count = 0;
                                s = ort_api->GetDimensionsCount(tensor_info, &dim_count);
                                // Allocate arrays for numeric dims and symbolic names
                                std::vector<int64_t> dims(dim_count);
                                std::vector<const char*> sym(dim_count, nullptr);
                                s = ort_api->GetDimensions(tensor_info, dims.data(), dim_count);
                                s = ort_api->GetSymbolicDimensions(tensor_info, sym.data(), dim_count);
                                // Read strings NOW while type_info_ptr is still alive
                                for (size_t d = 0; d < dim_count; d++) {
                                    if (sym[d] && sym[d][0] != '\0') {
                                        std::string name(sym[d]);  // copy before release
                                        std::cerr << "[Embedding] input[" << i << "] dim[" << d
                                            << "] = '" << name << "'" << std::endl;
                                        sym_dim_names.push_back(name);
                                    }
                                }
                            }
                            // Release AFTER we've copied all strings out
                            ort_api->ReleaseTypeInfo(type_info_ptr);
                        }
                    }
#if USE_COREML_FOR_EMBEDDINGS
                    auto override_dim = [&](const char* name, int64_t value) {
                        OrtStatus* s = ort_api->AddFreeDimensionOverrideByName(
                            session_options,  // implicit OrtSessionOptions* conversion
                            name,
                            value
                        );
                        if (s) {
                            std::cerr << "[CoreML] dim override failed for '" << name << "': "
                                << ort_api->GetErrorMessage(s) << std::endl;
                            ort_api->ReleaseStatus(s);
                        }
                        };
                    // ── 2. Apply overrides using the names we just found ─────────────────────
                    std::unordered_map<std::string, int64_t> dim_overrides;
                    for (const auto& name : sym_dim_names) {
                        if (dim_overrides.count(name)) continue;  // already seen
                        // Heuristic: first unique name is batch, second is sequence.
                        // Works for all standard encoder exports.
                        int64_t val = (dim_overrides.empty())
                            ? 1
                            : static_cast<int64_t>(max_position_embeddings);
                        dim_overrides[name] = val;
                    }
                    for (const auto& [name, value] : dim_overrides) {
                        override_dim(name.c_str(), value);
                        std::cerr << "[Embedding] override '" << name
                            << "' = " << value << std::endl;
                    }
                    embedding_coreml = true;
                    // CoreML: runs on ANE (Apple Neural Engine) + GPU on Apple Silicon.
                    // Falls back to CPU automatically for any unsupported ops.
                    std::unordered_map<std::string, std::string> coreml_opts;
                    coreml_opts["ModelCacheDirectory"] = (fs::path(embedding_model_path).parent_path() / "coreml_cache").string();
                    coreml_opts["MLComputeUnits"] = "ALL";
                    coreml_opts["ModelFormat"] = "MLProgram"; // Core ML 5+ (.mlpackage)
                    coreml_opts["RequireStaticInputShapes"] = "0";       // allow dynamic batch
                    coreml_opts["EnableOnSubgraphs"] = "0";
                    try {
                        session_options.AppendExecutionProvider("CoreML", coreml_opts);
                        embedding_fingerprint = get_system_fingerprint(embedding_model_path, "CoreML");
                        std::cerr << "[Embedding] CoreML EP loaded." << std::endl;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "[Embedding] CoreML EP unavailable, using CPU: "
                            << e.what() << std::endl;
                        embedding_fingerprint = get_system_fingerprint(embedding_model_path, "CPU");
                    }
#else
                    embedding_fingerprint = get_system_fingerprint(embedding_model_path, "CPU");
#endif

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
                    LoadModelConfig(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()),
                        cls_id_embeddings,
                        sep_id_embeddings,
                        max_position_embeddings,
                        ranking_mode_embeddings);
                    embeddings_tokenizer = LoadTokenizer(wchar_to_utf8(fs::path(embedding_model_path).parent_path().c_str()));
#else
                    LoadModelConfig(fs::path(embedding_model_path).parent_path(),
                        cls_id_embeddings,
                        sep_id_embeddings,
                        max_position_embeddings,
                        ranking_mode_embeddings);
                    embeddings_tokenizer = LoadTokenizer(fs::path(embedding_model_path).parent_path());
#endif
                    embedding_model_created = get_created_timestamp();
                }
                catch (const std::exception& e) {
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
    std::vector<int64_t> reranking_input_shape = { 1 }; // Batch size 1
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
                    LoadModelConfig(wchar_to_utf8(fs::path(reranker_model_path).parent_path().c_str()),
                        rerank_cls_id,
                        rerank_sep_id,
                        rerank_max_position_embeddings,
                        ranking_mode);
                    rerank_tokenizer = LoadTokenizer(wchar_to_utf8(fs::path(reranker_model_path).parent_path().c_str()));
#else
                    LoadModelConfig(fs::path(reranker_model_path).parent_path(),
                        rerank_cls_id,
                        rerank_sep_id,
                        rerank_max_position_embeddings,
                        ranking_mode);
                    rerank_tokenizer = LoadTokenizer(fs::path(reranker_model_path).parent_path());
#endif
                    reranking_model_created = get_created_timestamp();
                }
                catch (const std::exception& e) {
                    std::cerr << "Failed to load model: " << e.what() << std::endl;
                    return 1;
                }
            }
        }
    }

    std::unique_ptr<Ort::Session>                        tts_session;
    std::unique_ptr<Ort::Env>                            tts_env;
    std::string                                          tts_modelName;
    long long                                            tts_model_created = 0;
    std::vector<std::string>                             tts_input_node_names;
    std::vector<std::string>                             tts_output_node_names;
    std::vector<const char*>                             tts_input_names_c_array;
    std::vector<const char*>                             tts_output_names_c_array;
    size_t                                               num_tts_input_nodes = 0;
    size_t                                               num_tts_output_nodes = 0;
    std::unordered_map<std::string, KokoroVoice>         tts_voices;
    std::unordered_map<std::string, int64_t>             tts_vocab;
    bool                                                 tts_espeak_ready = false;

    if (tts_model_path.length() != 0) {
        if (fs::exists(tts_model_path) && fs::is_regular_file(tts_model_path)) {
            std::cerr << "[TTS] Loading from " << tts_model_path << std::endl;
            try {
                tts_env = std::make_unique<Ort::Env>(
                    ORT_LOGGING_LEVEL_WARNING, "TTS");

                std::string tts_dir =
                    fs::path(tts_model_path).parent_path().string();
                tts_modelName = get_model_name(tts_dir);

                Ort::SessionOptions sopts;
                sopts.SetIntraOpNumThreads(intra_op_threads);
                sopts.SetGraphOptimizationLevel(
                    GraphOptimizationLevel::ORT_ENABLE_ALL);
                sopts.AddConfigEntry("session.intra_op.allow_spinning", "0");

#ifdef WIN32
                tts_session = std::make_unique<Ort::Session>(*tts_env, tts_model_path_u16.c_str(), sopts);
#else
                tts_session = std::make_unique<Ort::Session>(*tts_env, tts_model_path.c_str(), sopts);
#endif

                Ort::AllocatorWithDefaultOptions tts_alloc;
                num_tts_input_nodes = tts_session->GetInputCount();
                num_tts_output_nodes = tts_session->GetOutputCount();

                for (size_t i = 0; i < num_tts_input_nodes; i++) {
                    auto p = tts_session->GetInputNameAllocated(i, tts_alloc);
                    tts_input_node_names.push_back(p.get());
                }
                for (size_t i = 0; i < num_tts_output_nodes; i++) {
                    auto p = tts_session->GetOutputNameAllocated(i, tts_alloc);
                    tts_output_node_names.push_back(p.get());
                }
                for (const auto& n : tts_input_node_names)
                    tts_input_names_c_array.push_back(n.c_str());
                for (const auto& n : tts_output_node_names)
                    tts_output_names_c_array.push_back(n.c_str());

                // Log actual input names so mismatches are immediately visible
                std::cout << "[TTS] Model inputs (" << num_tts_input_nodes << "):";
                for (const auto& n : tts_input_node_names)
                    std::cout << " '" << n << "'";
                std::cout << std::endl;

                tts_vocab = LoadKokoroVocab(tts_dir);
                tts_voices = LoadKokoroVoices(tts_dir);
                tts_espeak_ready = InitEspeak(tts_dir);
                tts_model_created = get_created_timestamp();

            }
            catch (const std::exception& e) {
                std::cerr << "[TTS] Failed to load: " << e.what() << std::endl;
                return 1;
            }
        }
    }

    //    const std::string instruction = "Given a web search query, retrieve relevant passages that answer the query";

        // ---------------------------------------------------------
        // SERVER MODE
        // ---------------------------------------------------------
    if (server_mode) {
        std::mutex inference_mutex;
        std::mutex tts_mutex;
        httplib::Server svr;

        // Route: /v1/chat/completions
        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(inference_mutex);
            std::cout << "[Server] /v1/chat/completions request received." << std::endl;

            try {

                if (model_created == 0) {
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
                bool has_tools = false;
                std::string tools_str = "";
                std::string guidance_string_type;
                std::string guidance_string;

                before_run_inference(req.body,
                    prompt,
                    max_tokens,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                    n,
                    is_stream,
                    has_tools,
                    tools_str,
                    *tokenizer.get(),
                    chat_template,
                    guidance_string_type,
                    guidance_string);

                if (is_stream) {
                    std::string req_id = get_openai_style_id();

                    // 1. Extract raw pointers safely.
                    // Since 'model' and 'tokenizer' live in main() for the lifetime of the app,
                    // these pointers will remain valid while the stream runs.
                    OgaModel* raw_model = model.get();
                    OgaTokenizer* raw_tokenizer = tokenizer.get();

                    // 2. Explicitly capture EVERYTHING by value (copy) or by safe pointer.
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
                            has_tools,
                            guidance_string_type,
                            guidance_string,
                            stop_tokens
                        ](size_t offset, httplib::DataSink& sink) {

                        const Json::UInt64 stream_created =
                            static_cast<Json::UInt64>(std::time(nullptr));

                        // Send initial role packet (optional but good practice)
                        for (int i = 0; i < n; i++) {
                            std::string role_chunk = create_stream_chunk(i, req_id, modelName, fingerprint, "", false, stream_created);
                            sink.write(role_chunk.data(), role_chunk.size());
                        }

                        // Define a callback to handle tokens as they are generated
                        auto token_callback = [&, stream_created](const std::string& token, unsigned int choice_index, bool is_tool) {
                            Json::Value root(Json::objectValue);
                            root["id"] = req_id;
                            root["object"] = "chat.completion.chunk";
                            root["created"] = stream_created;
                            root["model"] = modelName;
                            root["system_fingerprint"] = fingerprint;
                            Json::Value choices(Json::arrayValue);
                            Json::Value choice(Json::objectValue);
                            choice["index"] = choice_index;
                            Json::Value delta(Json::objectValue);

                            if (is_tool) {
                                std::vector<ParsedToolCall> tool_calls_parsed = parse_tool_call_json(token);
                                if (tool_calls_parsed.empty()) {
                                    is_tool = false;
                                }
                                else {
                                    delta["content"] = Json::nullValue;

                                    Json::Value tool_calls_node(Json::arrayValue);
                                    for (int tc_idx = 0; tc_idx < (int)tool_calls_parsed.size(); ++tc_idx) {
                                        Json::Value tc(Json::objectValue);
                                        tc["id"] = "call_" + get_openai_style_id();
                                        tc["type"] = "function";
                                        tc["index"] = tc_idx;
                                        Json::Value func(Json::objectValue);
                                        func["name"] = tool_calls_parsed[tc_idx].name;
                                        func["arguments"] = tool_calls_parsed[tc_idx].arguments;
                                        tc["function"] = func;
                                        tool_calls_node.append(tc);
                                    }

                                    delta["tool_calls"] = tool_calls_node;
                                    choice["finish_reason"] = "tool_calls";
                                }
                            }

                            if (!is_tool) {
                                delta["content"] = token;
                                choice["finish_reason"] = Json::nullValue;
                            }

                            choice["delta"] = delta;
                            choices.append(choice);
                            root["choices"] = choices;

                            Json::StreamWriterBuilder writer;
                            writer["indentation"] = "";
                            std::string chunk = "data: " + Json::writeString(writer, root) + "\n\n";

                            // Write immediately to the client
                            sink.write(chunk.data(), chunk.size());

                            return true; // Keep going
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
                            has_tools,
                            stop_tokens,
                            token_callback
                        );
                        // 4. Send finish reason
                        std::string finish_chunk = create_stream_chunk(n, req_id, modelName, fingerprint, "", true, stream_created);
                        sink.write(finish_chunk.data(), finish_chunk.size());

                        // 5. Send [DONE] to close the stream for the client
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.data(), done.size());

                        sink.done(); // Close the connection
                        return false;
                    }
                    );

                }
                else {
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
                        guidance_string,
                        has_tools,
                        stop_tokens
                    );
                    res.set_content(response_json, "application/json");
                    res.status = 200;
                }
            }
            catch (const std::exception& e) {
                std::string error_str = MakeErrorJson(e.what(), "invalid_request_error");
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
            if (model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = modelName;
                modelCard["object"] = "model";
                modelCard["created"] = model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            if (embedding_model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = embedding_modelName;
                modelCard["object"] = "model";
                modelCard["created"] = embedding_model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            if (reranking_model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = reranking_modelName;
                modelCard["object"] = "model";
                modelCard["created"] = reranking_model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            if (tts_model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = tts_modelName;
                modelCard["object"] = "model";
                modelCard["created"] = tts_model_created;
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

                if (reranking_model_created == 0) {
                    throw std::invalid_argument("[Rerank] Model not loaded.");
                }

                std::string query;
                int top_n = -1;
                std::vector<std::string> documents;
                before_run_reranking(req.body, query, top_n, documents);

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
                }
                else {
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
                                for (int x : q) { ids.push_back(x); }
                                ids.push_back(rerank_sep_id); // <sep>
                                for (int x : d) { ids.push_back(x); }
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
                                for (int x : q) { ids.push_back(x); type_ids.push_back(0); }
                                ids.push_back(rerank_sep_id); // [SEP]
                                type_ids.push_back(0);
                                for (int x : d) { ids.push_back(x); type_ids.push_back(1); }
                                ids.push_back(rerank_sep_id); // [SEP]
                                type_ids.push_back(1);
                                break;
                            case RERANKING_LLM:
                            default:
                                ids = rerank_tokenizer->Encode(query + "\n" + documents[i]);
                                break;
                            }

                            if (ids.size() > rerank_max_position_embeddings) {
                                ids.resize(rerank_max_position_embeddings - 1);
                                int end_token_id = rerank_sep_id;
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
            }
            catch (const std::exception& e) {
                std::string error_str = MakeErrorJson(e.what(), "invalid_request_error");
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
            });

        // Route: /v1/embeddings
        svr.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {

            std::cout << "[Server] /v1/embeddings request received." << std::endl;

            try {

                if (embedding_model_created == 0) {
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
                        sep_id_embeddings,
                        embedding_coreml);
                    break;
                }
                res.set_content(response_json, "application/json");
                res.status = 200;
            }
            catch (const std::exception& e) {
                std::string error_str = MakeErrorJson(e.what(), "invalid_request_error");
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
            });

        // Route: /v1/contextualizedembeddings
        auto contextualized_embeddings_handler = [&](const httplib::Request& req, httplib::Response& res) {

            std::cout << "[Server] /v1/contextualizedembeddings request received." << std::endl;

            try {

                if (embedding_model_created == 0) {
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
            }
            catch (const std::exception& e) {
                std::string error_str = MakeErrorJson(e.what(), "invalid_request_error");
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
            };

        svr.Post("/v1/contextualizedembeddings", contextualized_embeddings_handler);
        svr.Post("/v1/contextualized/embeddings", contextualized_embeddings_handler);

        svr.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(tts_mutex);

            std::cout << "[Server] /v1/audio/speech request received." << std::endl;

            try {
                if (tts_model_created == 0)
                    throw std::invalid_argument("[TTS] Model not loaded.");
                if (!tts_espeak_ready)
                    throw std::runtime_error("[TTS] espeak-ng not initialized.");
                if (tts_vocab.empty())
                    throw std::runtime_error("[TTS] vocab not loaded (missing config.json).");
                if (tts_voices.empty())
                    throw std::runtime_error("[TTS] No voices loaded (missing voices/*.bin).");

                // Parse request body
                Json::Value root;
                Json::CharReaderBuilder builder;
                std::string errors;
                std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
                if (!reader->parse(req.body.c_str(),
                    req.body.c_str() + req.body.size(),
                    &root, &errors) || !root.isObject())
                    throw std::invalid_argument("Invalid JSON body.");

                std::string input = root.get("input", "").asString();
                std::string voice = root.get("voice", "af_heart").asString();
                std::string fmt = root.get("response_format", "wav").asString();
                float       speed = root.get("speed", 1.0f).asFloat();

                if (input.empty())
                    throw std::invalid_argument("'input' field is required.");

                // Clamp speed
                speed = std::max(0.5f, std::min(2.0f, speed));

                // Resolve voice — fall back to first available
                auto voice_it = tts_voices.find(voice);
                if (voice_it == tts_voices.end()) {
                    voice_it = tts_voices.begin();
                    std::cerr << "[TTS] Voice '" << voice
                        << "' not found, using '" << voice_it->first
                        << "'." << std::endl;
                }

                std::vector<uint8_t> audio = run_tts(
                    tts_session.get(),
                    input,
                    voice_it->second,
                    speed,
                    tts_vocab,
                    tts_input_names_c_array, num_tts_input_nodes,
                    tts_output_names_c_array, num_tts_output_nodes);

                if (audio.empty())
                    throw std::runtime_error("TTS produced no audio output.");

                if (fmt == "pcm") {
                    // Raw int16 PCM — skip the 44-byte WAV header
                    res.set_content(
                        reinterpret_cast<const char*>(audio.data() + 44),
                        audio.size() - 44,
                        "application/octet-stream");
                }
                else {
                    res.set_content(
                        reinterpret_cast<const char*>(audio.data()),
                        audio.size(),
                        "audio/wav");
                }
                res.status = 200;

            }
            catch (const std::exception& e) {
                res.set_content(MakeErrorJson(e.what()), "application/json");
                res.status = 400;
                std::cerr << "[TTS] Error: " << e.what() << std::endl;
            }
            });

        std::cout << "[Server] Listening on " << host << ":" << port << std::endl;

        svr.new_task_queue = [] { return new httplib::ThreadPool(2); };
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
            FILE* f = _fopen(input_path, _rb);
            if (f) {
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

        std::string request_str((const char*)cli_request_json.data(), cli_request_json.size());
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
            bool has_tools = false;
            std::string tools_str = "";
            std::string guidance_string_type;
            std::string guidance_string;

            before_run_inference(request_str,
                prompt,
                max_tokens,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                n,
                is_stream,
                has_tools,
                tools_str,
                *tokenizer.get(),
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
                guidance_string,
                has_tools,
                stop_tokens
            );

        }
        catch (const std::exception& e) {
            response = MakeErrorJson(e.what(), "invalid_request_error");
        }

        // Output logic
        if (!output_path) {
            std::cout << response << std::endl;
        }
        else {
            FILE* f = _fopen(output_path, _wb);
            if (f) {
                fwrite(response.c_str(), 1, response.length(), f);
                fclose(f);
            }
        }
    }

    OgaShutdown();

    return 0;
}
