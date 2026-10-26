#ifndef ONNX_SYSTEMONE_RUNTIME_H
#define ONNX_SYSTEMONE_RUNTIME_H

#include "systemone.h"
#include "tokenizers_cpp.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>

namespace systemone {

inline std::string path_utf8(const std::filesystem::path& path) {
    const auto bytes = path.u8string();
    return std::string(bytes.begin(), bytes.end());
}

inline std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open decision model file: " + path_utf8(path));
    std::ostringstream content;
    content << file.rdbuf();
    if (file.bad()) throw std::runtime_error("Could not read decision model file: " + path_utf8(path));
    return content.str();
}

inline std::string escape_controls(const std::string& text) {
    static const std::regex control("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(text, control, "<\xC2\xA6$1\xC2\xA6>");
}

inline std::vector<std::string> input_names(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    for (size_t i = 0; i < session.GetInputCount(); ++i)
        names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    return names;
}

inline double tensor_score(const Ort::Value& tensor, size_t index) {
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    if (index >= info.GetElementCount()) throw std::runtime_error("Decision output index is out of bounds.");
    switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return tensor.GetTensorData<float>()[index];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return tensor.GetTensorData<Ort::Float16_t>()[index].ToFloat();
    default:
        throw std::runtime_error("Decision model output must be float32 or float16.");
    }
}

struct KevInput {
    std::vector<int64_t> ids;
    std::vector<std::vector<size_t>> option_ends;
};

struct D1Input {
    std::vector<int64_t> ids;
    std::vector<std::vector<int32_t>> readout;
};

class Model {
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "SystemOne"};
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Session> embedding_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    Json::Value config_;
    std::filesystem::path root_;
    std::string id_;
    bool kev_ = false;
    size_t context_limit_ = max_sequence_tokens;
    std::array<int64_t, 5> delimiters_{};

    std::vector<int64_t> encode(const std::string& text, bool caller = true) {
        const auto tokens = tokenizer_->Encode(caller ? escape_controls(text) : text);
        return std::vector<int64_t>(tokens.begin(), tokens.end());
    }

    static void append(std::vector<int64_t>& target, const std::vector<int64_t>& source) {
        target.insert(target.end(), source.begin(), source.end());
    }

    static std::string option_text(const std::string& key, const Json::Value& desc) {
        return desc.isNull() || (desc.isString() && desc.asString().empty())
            ? key : key + ": " + kev_render(desc);
    }

    static size_t positive_limit(const Json::Value& value, const std::string& field) {
        if (!value.isUInt64() || value.asUInt64() == 0 ||
            value.asUInt64() > static_cast<Json::UInt64>(std::numeric_limits<int64_t>::max()))
            throw std::runtime_error("Invalid decision model limit: " + field);
        return static_cast<size_t>(value.asUInt64());
    }

    void validate_graphs() {
        const auto names = input_names(*session_);
        if (kev_) {
            if (std::set<std::string>(names.begin(), names.end()) !=
                std::set<std::string>{"input_ids", "attention_mask"})
                throw std::runtime_error("Kev graph must take input_ids and attention_mask.");
        } else {
            bool embeds = false, mask = false;
            for (const auto& name : names) {
                if (name == "inputs_embeds") embeds = true;
                else if (name == "attention_mask") mask = true;
                else if (name.rfind("past_conv", 0) != 0 && name.rfind("past_key_values", 0) != 0)
                    throw std::runtime_error("Unsupported d1 decoder input: " + name);
            }
            if (!embeds || !mask) throw std::runtime_error("d1 decoder lacks inputs_embeds or attention_mask.");
            if (input_names(*embedding_) != std::vector<std::string>{"input_ids"} || embedding_->GetOutputCount() != 1)
                throw std::runtime_error("Unsupported d1 embedding graph.");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        bool logits = false;
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
            if (std::string(session_->GetOutputNameAllocated(i, allocator).get()) == "logits") logits = true;
        if (!logits) throw std::runtime_error("Decision graph has no logits output.");
    }

    std::vector<int32_t> single_tokens(const std::vector<std::string>& forms) {
        std::vector<int32_t> ids;
        for (const auto& form : forms) {
            const auto encoded = tokenizer_->Encode(form);
            if (encoded.size() == 1 && std::find(ids.begin(), ids.end(), encoded[0]) == ids.end())
                ids.push_back(encoded[0]);
        }
        return ids;
    }

    std::vector<std::pair<std::string, int32_t>> aliases(const Question& q) {
        bool letters = true;
        for (const auto& key : q.keys)
            letters = letters && key.size() == 1 &&
                ((key[0] >= 'A' && key[0] <= 'Z') || (key[0] >= 'a' && key[0] <= 'z'));
        std::vector<std::string> pool;
        for (char c = 'A'; c <= 'Z'; ++c) pool.emplace_back(1, c);
        for (int i = 0; i < 100; ++i) pool.push_back((i < 10 ? "0" : "") + std::to_string(i));
        for (char c = 'a'; c <= 'z'; ++c) pool.emplace_back(1, c);
        for (int i = 0; i < 200; ++i) pool.push_back("#" + std::to_string(i));
        for (char a = 'A'; a <= 'Z'; ++a)
            for (char b = 'A'; b <= 'Z'; ++b) pool.push_back(std::string{a, b});
        std::set<int32_t> used;
        std::vector<std::pair<std::string, int32_t>> result;
        auto take = [&](const std::string& code) {
            const auto tokens = tokenizer_->Encode(code);
            if (tokens.size() != 1 || !used.insert(tokens[0]).second) return false;
            result.emplace_back(code, tokens[0]);
            return true;
        };
        for (size_t i = 0; i < q.keys.size(); ++i) {
            const auto code = letters ? q.keys[i] : q.keys.size() <= 26
                ? std::string(1, static_cast<char>('A' + i))
                : (i < 10 ? "0" : "") + std::to_string(i);
            if (take(code)) continue;
            bool found = false;
            for (const auto& fallback : pool) {
                if (take(fallback)) { found = true; break; }
            }
            if (!found) throw ValidationError({"questions", q.id, "criteria"}, "No distinct single-token option code available.");
        }
        return result;
    }

    static std::string d1_content(const Json::Value& value) {
        if (value.isNull()) return "";
        return value.isString() ? value.asString() : ordered_json(value);
    }

    std::vector<std::vector<double>> run_kev(KevInput& input) {
        const int64_t length = static_cast<int64_t>(input.ids.size());
        const std::array<int64_t, 2> shape{1, length};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> mask(input.ids.size(), 1);
        std::array<Ort::Value, 2> tensors{
            Ort::Value::CreateTensor<int64_t>(memory, input.ids.data(), input.ids.size(), shape.data(), shape.size()),
            Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), shape.data(), shape.size())
        };
        const char* names[]{"input_ids", "attention_mask"};
        const char* output_names[]{"logits"};
        auto output = session_->Run(Ort::RunOptions{nullptr}, names, tensors.data(), 2, output_names, 1);
        if (output[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, length})
            throw std::runtime_error("Kev logits must have shape [1, sequence].");
        std::vector<std::vector<double>> probabilities;
        for (const auto& ends : input.option_ends) {
            std::vector<double> scores;
            for (size_t index : ends) scores.push_back(tensor_score(output[0], index));
            probabilities.push_back(softmax(scores));
        }
        return probabilities;
    }

    std::vector<double> run_d1(D1Input& input) {
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(input.ids.size())};
        auto token_tensor = Ort::Value::CreateTensor<int64_t>(
            memory, input.ids.data(), input.ids.size(), shape.data(), shape.size());
        const char* embed_input[]{"input_ids"};
        auto embed_output = embedding_->GetOutputNameAllocated(0, allocator);
        const char* embed_names[]{embed_output.get()};
        auto embeddings = embedding_->Run(Ort::RunOptions{nullptr}, embed_input, &token_tensor, 1, embed_names, 1);
        const auto embed_info = embeddings[0].GetTensorTypeAndShapeInfo();
        const auto embed_shape = embed_info.GetShape();
        if (embed_shape.size() != 3 || embed_shape[0] != 1 || embed_shape[1] != shape[1])
            throw std::runtime_error("d1 embeddings must have shape [1, sequence, hidden].");
        std::vector<int64_t> mask(input.ids.size(), 1);
        const auto names = input_names(*session_);
        std::vector<const char*> raw_names;
        std::vector<Ort::Value> tensors;
        for (size_t i = 0; i < names.size(); ++i) {
            raw_names.push_back(names[i].c_str());
            auto type_info = session_->GetInputTypeInfo(i);
            const auto info = type_info.GetTensorTypeAndShapeInfo();
            const auto type = info.GetElementType();
            if (names[i] == "inputs_embeds") {
                if (type != embed_info.GetElementType())
                    throw std::runtime_error("d1 embedding and decoder precision do not match.");
                tensors.push_back(std::move(embeddings[0]));
            } else if (names[i] == "attention_mask") {
                if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                    throw std::runtime_error("d1 attention_mask must be int64.");
                tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory, mask.data(), mask.size(), shape.data(), shape.size()));
            } else {
                auto cache_shape = info.GetShape();
                const auto& text_config = config_["text_config"];
                if (names[i].rfind("past_conv", 0) == 0) {
                    cache_shape = {1, static_cast<int64_t>(positive_limit(text_config["conv_dim"], "conv_dim")),
                        static_cast<int64_t>(positive_limit(text_config["conv_L_cache"], "conv_L_cache"))};
                } else {
                    const auto heads = positive_limit(text_config["num_attention_heads"], "num_attention_heads");
                    const auto hidden = positive_limit(text_config["hidden_size"], "hidden_size");
                    if (hidden % heads) throw std::runtime_error("Invalid d1 attention head dimension.");
                    cache_shape = {1, static_cast<int64_t>(positive_limit(text_config["num_key_value_heads"], "num_key_value_heads")),
                        0, static_cast<int64_t>(hidden / heads)};
                }
                if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                    throw std::runtime_error("Unsupported d1 cache precision.");
                auto cache = Ort::Value::CreateTensor(allocator, cache_shape.data(), cache_shape.size(), type);
                const auto count = cache.GetTensorTypeAndShapeInfo().GetElementCount();
                if (count) std::memset(cache.GetTensorMutableRawData(), 0, count * (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 2));
                tensors.push_back(std::move(cache));
            }
        }
        const char* output_names[]{"logits"};
        auto output = session_->Run(Ort::RunOptions{nullptr}, raw_names.data(), tensors.data(), tensors.size(), output_names, 1);
        const auto output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 1 || output_shape[2] <= 0)
            throw std::runtime_error("d1 logits must have shape [1, 1, vocabulary].");
        std::vector<double> scores;
        for (const auto& group : input.readout) {
            double best = -std::numeric_limits<double>::infinity();
            for (int32_t token : group) {
                if (token < 0 || token >= output_shape[2])
                    throw std::runtime_error("d1 readout token is outside the output vocabulary.");
                const double score = tensor_score(output[0], static_cast<size_t>(token));
                if (!std::isfinite(score)) throw std::runtime_error("d1 returned a non-finite readout score.");
                best = std::max(best, score);
            }
            scores.push_back(best);
        }
        return softmax(scores);
    }

public:
    Model(const std::filesystem::path& graph, int threads) {
        if (!std::filesystem::is_regular_file(graph) || graph.extension() != ".onnx")
            throw std::runtime_error("-D requires a decision model .onnx file.");
        root_ = graph.parent_path();
        if (!std::filesystem::exists(root_ / "config.json") && root_.filename() == "onnx") root_ = root_.parent_path();
        if (root_.empty()) root_ = ".";
        config_ = parse_json(read_file(root_ / "config.json"));
        if (!config_.isObject() || config_.isMember("kev") == config_.isMember("d1"))
            throw std::runtime_error("Decision config.json must identify exactly one backend: kev or d1.");
        kev_ = config_.isMember("kev");
        if (!(kev_ ? config_["kev"] : config_["d1"]).isObject())
            throw std::runtime_error("Invalid decision backend metadata.");
        auto absolute_root = std::filesystem::absolute(root_).lexically_normal();
        if (absolute_root.filename().empty()) absolute_root = absolute_root.parent_path();
        id_ = path_utf8(absolute_root.filename());
        if (id_.empty()) throw std::runtime_error("Decision model directory must have a non-empty name.");
        auto tokenizer_config = parse_json(read_file(root_ / "tokenizer.json"));
        tokenizer_config["post_processor"] = Json::nullValue;
        tokenizer_config["truncation"] = Json::nullValue;
        tokenizer_config["padding"] = Json::nullValue;
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_string(tokenizer_config));
        if (!tokenizer_) throw std::runtime_error("Could not load decision tokenizer.");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.DisableMemPattern();
        options.DisableCpuMemArena();
        if (kev_) {
            const auto& metadata = config_["kev"];
            const std::array<std::string, 5> keys{"state", "question", "option_start", "option_end", "decide"};
            std::set<int64_t> seen;
            for (size_t i = 0; i < keys.size(); ++i) {
                const auto& value = metadata["delimiter_ids"][keys[i]];
                if (!value.isInt64() || value.asInt64() < 0 || !seen.insert(value.asInt64()).second)
                    throw std::runtime_error("Invalid Kev delimiter id: " + keys[i]);
                delimiters_[i] = value.asInt64();
                if (!metadata["delimiters"][keys[i]].isString() ||
                    tokenizer_->TokenToId(metadata["delimiters"][keys[i]].asString()) != delimiters_[i])
                    throw std::runtime_error("Kev tokenizer and delimiter configuration do not match.");
            }
            context_limit_ = std::min(max_sequence_tokens, positive_limit(metadata["max_branch_tokens"], "max_branch_tokens"));
        } else {
            context_limit_ = std::min(max_sequence_tokens,
                positive_limit(config_["text_config"]["max_position_embeddings"], "max_position_embeddings"));
            const std::string stem = path_utf8(graph.stem());
            const std::string prefix = "decoder_model_merged";
            if (stem.rfind(prefix, 0) != 0)
                throw std::runtime_error("d1 requires decoder_model_merged[_quantized|_q4].onnx.");
            const auto embedding_path = graph.parent_path() / ("embed_tokens" + stem.substr(prefix.size()) + ".onnx");
            embedding_ = std::make_unique<Ort::Session>(env_, embedding_path.c_str(), options);
        }
        session_ = std::make_unique<Ort::Session>(env_, graph.c_str(), options);
        validate_graphs();
    }

    const std::string& id() const { return id_; }
    const char* backend() const { return kev_ ? "kev" : "d1"; }

    KevInput prepare_kev(const Request& request) {
        KevInput input;
        input.ids.push_back(delimiters_[0]);
        append(input.ids, encode(kev_render(request.state)));
        const size_t state_length = input.ids.size();
        const size_t state_limit = std::min(max_sequence_tokens,
            positive_limit(config_["kev"]["max_state_tokens"], "max_state_tokens"));
        if (state_length > state_limit)
            throw ValidationError({"state"}, "Kev state exceeds " + std::to_string(state_limit) + " tokens.");
        const auto option_limit = std::min(size_t{255}, positive_limit(config_["kev"]["max_options"], "max_options"));
        for (const auto& q : request.questions) {
            if (q.keys.size() > option_limit)
                throw ValidationError({"questions", q.id, "criteria"}, "Too many options for this Kev model.");
            const size_t base = input.ids.size();
            input.ids.push_back(delimiters_[1]);
            append(input.ids, encode(kev_render(q.instructions)));
            std::vector<std::string> options;
            if (q.type == "noul") options = {option_text("no", q.criteria["false"]), option_text("yes", q.criteria["true"])};
            else if (q.type == "choice") {
                for (const auto& key : q.keys) options.push_back(option_text(key, q.criteria[key]));
            } else {
                for (const auto& level : q.criteria) options.push_back(kev_render(level));
            }
            std::vector<size_t> ends;
            for (const auto& option : options) {
                input.ids.push_back(delimiters_[2]);
                append(input.ids, encode(option));
                ends.push_back(input.ids.size());
                input.ids.push_back(delimiters_[3]);
            }
            input.ids.push_back(delimiters_[4]);
            if (state_length + input.ids.size() - base > context_limit_)
                throw ValidationError({"questions", q.id}, "Kev state plus question exceeds the model's row token limit.");
            if (input.ids.size() > max_sequence_tokens)
                throw ValidationError({"questions"}, "Packed Kev request exceeds 8192 tokens; split the questions across requests.");
            input.option_ends.push_back(std::move(ends));
        }
        return input;
    }

    D1Input prepare_d1(const Request& request, const Question& q) {
        D1Input input;
        std::string instruction = d1_content(q.instructions);
        std::string question;
        if (q.type == "noul") {
            question = instruction;
            if (q.criteria.isObject() && !q.criteria.empty())
                question += "\nYes: " + (q.criteria["true"].isNull() ? "None" : d1_content(q.criteria["true"])) +
                    "\nNo: " + (q.criteria["false"].isNull() ? "None" : d1_content(q.criteria["false"]));
            question += "\n\nReply with yes or no only.";
            // Keep shared answer construction in false/true order; d1's prompt uses yes/no.
            input.readout = {single_tokens({"no", "No", "NO"}), single_tokens({"yes", "Yes", "YES"})};
        } else if (q.type == "score") {
            question = instruction + "\n\n";
            for (size_t i = 0; i < q.keys.size(); ++i) {
                if (i) question += "\n";
                question += std::to_string(i) + " " + d1_content(q.criteria[static_cast<Json::ArrayIndex>(i)]);
                input.readout.push_back(single_tokens({std::to_string(i)}));
            }
            question += "\n\nReply with a single digit 0-" + std::to_string(q.keys.size() - 1) + " only.";
        } else {
            question = instruction + "\n\nOptions:\n";
            const auto codes = aliases(q);
            for (size_t i = 0; i < q.keys.size(); ++i) {
                if (i) question += "\n";
                std::string desc = d1_content(q.criteria[q.keys[i]]);
                if (desc.empty()) {
                    desc = q.keys[i];
                    std::replace(desc.begin(), desc.end(), '_', ' ');
                }
                question += codes[i].first + " " + desc;
                auto group = single_tokens({codes[i].first, " " + codes[i].first});
                input.readout.push_back(std::move(group));
            }
            question += "\n\nReply with the option code only.";
        }
        for (const auto& group : input.readout)
            if (group.empty())
                throw ValidationError({"questions", q.id}, "d1 requires single-token answer labels.");
        std::string body = escape_controls(d1_content(request.state)) + "\n\n\nQUESTION:\n" + escape_controls(question);
        if (body.find("<image>") != std::string::npos)
            throw ValidationError({"state"}, "Image markers are unsupported by the text-only d1 adapter.");
        const auto prompt = "<|startoftext|><|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n";
        input.ids = encode(prompt, false);
        if (input.ids.size() > context_limit_)
            throw ValidationError({"questions", q.id}, "d1 prompt exceeds " + std::to_string(context_limit_) + " tokens.");
        return input;
    }

    Json::Value evaluate(const Request& request) {
        if (request.model != id_ && request.model != (kev_ ? "kev-latest" : "d1-latest"))
            throw ValidationError({"model"}, "Requested model is not loaded. Use " + id_ + " or " + backend() + "-latest.");
        Json::Value result(Json::objectValue);
        result["model"] = id_;
        result["answers"] = Json::Value(Json::objectValue);
        size_t tokens = 0;
        std::vector<std::vector<double>> probabilities;
        if (kev_) {
            auto input = prepare_kev(request);
            tokens = input.ids.size();
            probabilities = run_kev(input);
        } else {
            std::vector<D1Input> inputs;
            for (const auto& q : request.questions) {
                inputs.push_back(prepare_d1(request, q));
                tokens += inputs.back().ids.size();
                if (tokens > max_request_tokens)
                    throw ValidationError({"questions"}, "d1 request exceeds 32768 total prompt tokens; split the questions.");
            }
            for (auto& input : inputs) probabilities.push_back(run_d1(input));
        }
        for (size_t i = 0; i < request.questions.size(); ++i)
            result["answers"][request.questions[i].id] = answer(request.questions[i], probabilities[i]);
        result["usage"]["input_tokens"] = Json::UInt64(tokens);
        result["usage"]["output_tokens"] = 0;
        return result;
    }
};

} // namespace systemone
#endif
