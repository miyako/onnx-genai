#ifndef ONNX_SYSTEMONE_LFM3_H
#define ONNX_SYSTEMONE_LFM3_H

// LiquidAI's own d1 ONNX exports (d1-2B-ONNX, d1-Omni-2B-ONNX; LFM3 backbone), text only.
//
// One decoder pass reads a state and up to `d1_set_size` questions packed after it
// ("set" attention: the graph masks each question from the others through `question_ids`).
// The graph returns logits only at the answer slots (`slot_positions`), so no KV or
// convolution cache is involved. The prompt, option codes, readout and temperature buckets
// follow the checkpoint's published reference code (prompt.py, attention.py, calibration.py).

#include "systemone.h"
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace systemone {

struct Lfm3Config {
    size_t set_size = 4;      // d1_set_size; the reference default when a checkpoint records none
    int codes = 1;            // d1_codes: option code scheme beyond 26 options
    std::string bos;          // tokenizer_config.json bos_token
    size_t hidden = 0;        // text_config.hidden_size: width of the (empty) media feature inputs
    double default_temperature = 1;
    std::map<std::string, double> temperatures;
    std::vector<std::string> specials;  // special added tokens, longest first

    static double positive_temperature(const Json::Value& value, const std::string& field) {
        if (!value.isNumeric() || !std::isfinite(value.asDouble()) || value.asDouble() <= 0)
            throw std::runtime_error("Invalid d1 calibration temperature: " + field);
        return value.asDouble();
    }

    // config: config.json; calibration: calibration.json or null; tokenizer: tokenizer.json;
    // tokenizer_config: tokenizer_config.json or null.
    static Lfm3Config load(const Json::Value& config, const Json::Value& calibration,
                           const Json::Value& tokenizer, const Json::Value& tokenizer_config) {
        Lfm3Config result;
        if (!config["d1_attention"].isString() || config["d1_attention"].asString() != "set")
            throw std::runtime_error("Only d1 checkpoints trained with \"set\" attention are supported.");
        if (config.isMember("d1_set_size")) {
            const auto& size = config["d1_set_size"];
            if (!size.isUInt() || size.asUInt() < 2 || size.asUInt() > 128)
                throw std::runtime_error("d1_set_size must be a whole number from 2 to 128.");
            result.set_size = size.asUInt();
        }
        if (config.isMember("d1_codes")) {
            const auto& codes = config["d1_codes"];
            if (!codes.isInt() || (codes.asInt() != 1 && codes.asInt() != 2))
                throw std::runtime_error("d1_codes must be 1 or 2.");
            result.codes = codes.asInt();
        }
        if (config.isMember("d1_prompt")) {
            const auto& prompt = config["d1_prompt"];
            if (!prompt.isInt() || (prompt.asInt() != 1 && prompt.asInt() != 2))
                throw std::runtime_error("d1_prompt must be 1 or 2.");
        }
        const auto& hidden = config["text_config"]["hidden_size"];
        if (!hidden.isUInt() || hidden.asUInt() == 0)
            throw std::runtime_error("d1 config lacks text_config.hidden_size.");
        result.hidden = hidden.asUInt();

        if (!calibration.isNull()) {
            if (!calibration.isObject()) throw std::runtime_error("calibration.json must be an object.");
            if (calibration.isMember("default"))
                result.default_temperature = positive_temperature(calibration["default"], "default");
            const auto& table = calibration["temperatures"];
            if (!table.isNull()) {
                if (!table.isObject()) throw std::runtime_error("calibration.json temperatures must be an object.");
                for (const auto& key : table.getMemberNames())
                    result.temperatures[key] = positive_temperature(table[key], key);
            }
        }

        if (tokenizer_config.isObject()) {
            const auto& bos = tokenizer_config["bos_token"];
            if (bos.isString()) result.bos = bos.asString();
            else if (bos.isObject() && bos["content"].isString()) result.bos = bos["content"].asString();
        }
        std::set<std::string> specials;
        for (const auto& token : tokenizer["added_tokens"])
            if (token["special"].asBool() && token["content"].isString() && token["content"].asString().size() > 1)
                specials.insert(token["content"].asString());
        result.specials.assign(specials.begin(), specials.end());
        std::stable_sort(result.specials.begin(), result.specials.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
        return result;
    }

    double temperature(const std::string& bucket) const {
        const auto found = temperatures.find(bucket);
        return found == temperatures.end() ? default_temperature : found->second;
    }
};

// calibration.py `bucket`
inline std::string lfm3_bucket(const std::string& type, size_t options) {
    if (type == "noul") return "noul";
    if (type == "score") return "score:" + std::to_string(options);
    if (options <= 3) return "choice:small";
    if (options <= 8) return "choice:mid";
    if (options <= 26) return "choice:large";
    return "choice:inventory";
}

// prompt.py `plain`: a zero-width space after the first character of every special token
// in caller text, repeated until none is left (a break can complete an overlapping one).
inline std::string lfm3_plain(std::string text, const std::vector<std::string>& specials) {
    static const std::string zwsp = "\xE2\x80\x8B";
    for (bool changed = true; changed;) {
        changed = false;
        std::string out;
        out.reserve(text.size());
        for (size_t i = 0; i < text.size();) {
            const std::string* hit = nullptr;
            for (const auto& special : specials)
                if (text.compare(i, special.size(), special) == 0) { hit = &special; break; }
            if (!hit) { out += text[i++]; continue; }
            out += hit->substr(0, 1) + zwsp + hit->substr(1);
            i += hit->size();
            changed = true;
        }
        text = std::move(out);
    }
    return text;
}

// prompt.py `option_codes` (single ASCII letter labels keep themselves).
inline std::vector<std::string> lfm3_codes(const std::vector<std::string>& labels, int scheme) {
    auto strip = [](const std::string& s) {
        const auto a = s.find_first_not_of(" \t\r\n\f\v");
        if (a == std::string::npos) return std::string();
        return s.substr(a, s.find_last_not_of(" \t\r\n\f\v") - a + 1);
    };
    bool letters = !labels.empty();
    std::vector<std::string> stripped;
    for (const auto& label : labels) {
        stripped.push_back(strip(label));
        const auto& s = stripped.back();
        letters = letters && s.size() == 1 && ((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z'));
    }
    if (letters) return stripped;
    std::vector<std::string> codes;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels.size() <= 26) codes.emplace_back(1, static_cast<char>('A' + i));
        else if (scheme == 2)
            codes.push_back(i < 26 ? std::string(1, static_cast<char>('A' + i))
                : std::string{static_cast<char>('A' + (i - 26) / 26), static_cast<char>('A' + (i - 26) % 26)});
        else codes.push_back((i < 10 ? "0" : "") + std::to_string(i));
    }
    return codes;
}

// prompt.py `_FALLBACK_POOL`
inline const std::vector<std::string>& lfm3_fallback_pool() {
    static const std::vector<std::string> pool = [] {
        std::vector<std::string> p;
        for (char c = 'A'; c <= 'Z'; ++c) p.emplace_back(1, c);
        for (int i = 0; i < 100; ++i) p.push_back((i < 10 ? "0" : "") + std::to_string(i));
        for (char c = 'a'; c <= 'z'; ++c) p.emplace_back(1, c);
        for (int i = 0; i < 200; ++i) p.push_back("#" + std::to_string(i));
        for (char a = 'A'; a <= 'Z'; ++a)
            for (char b = 'A'; b <= 'Z'; ++b) p.push_back(std::string{a, b});
        return p;
    }();
    return pool;
}

using Lfm3Encode = std::function<std::vector<int64_t>(const std::string&)>;

struct Lfm3Question {
    std::string suffix;                       // question block through the assistant lead
    std::vector<std::vector<int64_t>> groups; // readout tokens per option, onnx-genai key order
    std::string bucket;                       // calibration bucket
};

struct Lfm3Row {
    std::vector<int64_t> ids;
    std::vector<int64_t> question_ids;  // -1 on the state, k on its k-th question
    std::vector<int64_t> slots;         // each question's answer position
    std::vector<size_t> questions;      // request question indices, in slot order
};

inline std::string lfm3_content(const Json::Value& value) {
    if (value.isNull()) return "";
    return value.isString() ? value.asString() : ordered_json(value);
}

// Python's f"{value}" for a noul criterion: None when absent.
inline std::string lfm3_said(const Json::Value& value) {
    return value.isNull() ? "None" : lfm3_content(value);
}

inline std::vector<int64_t> lfm3_single(const Lfm3Encode& encode, const std::vector<std::string>& forms) {
    std::vector<int64_t> ids;
    for (const auto& form : forms) {
        const auto encoded = encode(form);
        if (encoded.size() == 1 && std::find(ids.begin(), ids.end(), encoded[0]) == ids.end())
            ids.push_back(encoded[0]);
    }
    return ids;
}

// prompt.py `question_block`, `suffix_layout` and `readout_ids` for noul, choice and score.
inline Lfm3Question lfm3_question(const Question& q, const Lfm3Config& config, const Lfm3Encode& encode) {
    static const std::string lead = "<think></think>";
    Lfm3Question result;
    const std::string said = lfm3_content(q.instructions);
    std::string body;
    if (q.type == "noul") {
        body = said;
        if (q.criteria.isObject() && !q.criteria.empty())
            body += "\nYes: " + lfm3_said(q.criteria["true"]) + "\nNo: " + lfm3_said(q.criteria["false"]);
        body += "\n\nReply with yes or no only.";
        result.groups = {lfm3_single(encode, {"no", "No", "NO"}), lfm3_single(encode, {"yes", "Yes", "YES"})};
        result.bucket = lfm3_bucket("noul", 2);
    } else if (q.type == "score") {
        body = said + "\n\n";
        for (size_t i = 0; i < q.keys.size(); ++i) {
            if (i) body += "\n";
            body += std::to_string(i) + " " + lfm3_content(q.criteria[static_cast<Json::ArrayIndex>(i)]);
            result.groups.push_back(lfm3_single(encode, {std::to_string(i)}));
        }
        body += "\n\nReply with a single digit 0-" + std::to_string(q.keys.size() - 1) + " only.";
        result.bucket = lfm3_bucket("score", q.keys.size());
    } else {
        body = said + "\n\nOptions:\n";
        const auto codes = lfm3_codes(q.keys, config.codes);
        std::set<int64_t> used;
        for (size_t i = 0; i < q.keys.size(); ++i) {
            std::string code;
            int64_t token = -1;
            auto take = [&](const std::string& raw) {
                const auto encoded = encode(raw);
                if (encoded.size() != 1 || !used.insert(encoded[0]).second) return false;
                code = raw;
                token = encoded[0];
                return true;
            };
            if (!take(codes[i])) {
                bool found = false;
                for (const auto& fallback : lfm3_fallback_pool())
                    if (take(fallback)) { found = true; break; }
                if (!found)
                    throw ValidationError({"questions", q.id, "criteria"}, "No distinct single-token option code available.");
            }
            std::string desc = lfm3_content(q.criteria[q.keys[i]]);
            if (desc.empty()) {
                desc = q.keys[i];
                std::replace(desc.begin(), desc.end(), '_', ' ');
            }
            if (i) body += "\n";
            body += code + " " + desc;
            std::vector<int64_t> group{token};
            for (auto id : lfm3_single(encode, {" " + code}))
                if (id != token) group.push_back(id);
            result.groups.push_back(std::move(group));
        }
        body += "\n\nReply with the option code only.";
        result.bucket = lfm3_bucket("choice", q.keys.size());
    }
    for (const auto& group : result.groups)
        if (group.empty())
            throw ValidationError({"questions", q.id}, "d1 requires single-token answer labels.");
    result.suffix = lfm3_plain(body, config.specials) + "<|im_end|>\n<|im_start|>assistant\n" + lead;
    return result;
}

// prompt.py `prefix_text` (no system turn, json_only state style).
inline std::string lfm3_prefix(const Json::Value& state, const Lfm3Config& config) {
    const std::string block = (state.isString() ? state.asString() : ordered_json(state)) + "\n\n";
    return config.bos + "<|im_start|>user\n" + lfm3_plain(block, config.specials) + "\nQUESTION:\n";
}

// attention.py `pack_set`: the first question tokenized with the prefix, each further one as
// "QUESTION:\n" + its suffix. Up to set_size questions a row, fewer when a row would pass
// `row_limit` tokens.
inline std::vector<Lfm3Row> lfm3_rows(const Request& request, const std::vector<Lfm3Question>& questions,
                                      const Lfm3Config& config, const Lfm3Encode& encode, size_t row_limit) {
    const auto prefix = lfm3_prefix(request.state, config);
    std::vector<Lfm3Row> rows;
    for (size_t i = 0; i < questions.size();) {
        Lfm3Row row;
        row.ids = encode(prefix + questions[i].suffix);
        const auto own = encode(questions[i].suffix).size();
        if (own > row.ids.size()) throw std::runtime_error("d1 question does not end the prompt on a token boundary.");
        const size_t state = row.ids.size() - own;
        if (row.ids.size() > row_limit)
            throw ValidationError({"questions", request.questions[i].id},
                "d1 prompt exceeds " + std::to_string(row_limit) + " tokens.");
        row.question_ids.assign(state, -1);
        row.question_ids.resize(row.ids.size(), 1);
        row.slots.push_back(static_cast<int64_t>(row.ids.size()) - 1);
        row.questions.push_back(i++);
        while (i < questions.size() && row.questions.size() < config.set_size) {
            const auto next = encode("QUESTION:\n" + questions[i].suffix);
            if (row.ids.size() + next.size() > row_limit) break;
            row.ids.insert(row.ids.end(), next.begin(), next.end());
            row.question_ids.resize(row.ids.size(), static_cast<int64_t>(row.questions.size() + 1));
            row.slots.push_back(static_cast<int64_t>(row.ids.size()) - 1);
            row.questions.push_back(i++);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

} // namespace systemone
#endif
