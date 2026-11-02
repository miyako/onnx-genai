#ifndef ONNX_SYSTEMONE_STRANDS_H
#define ONNX_SYSTEMONE_STRANDS_H

#include "systemone.h"
#include "tokenizers_cpp.h"
#include <charconv>
#include <limits>
#include <map>
#include <utility>

namespace systemone {

struct StrandsConfig {
    size_t context;
    std::map<std::string, double> temperatures;

    explicit StrandsConfig(const Json::Value& metadata) {
        const auto& length = metadata["max_length"];
        if (!length.isUInt64() || length.asUInt64() == 0 ||
            length.asUInt64() > static_cast<Json::UInt64>(std::numeric_limits<int64_t>::max()))
            throw std::runtime_error("Invalid Strands decider.max_length.");
        context = std::min(max_sequence_tokens, static_cast<size_t>(length.asUInt64()));
        const auto& by_kind = metadata["temperature_by_kind"];
        if (!by_kind.isNull() && !by_kind.isObject())
            throw std::runtime_error("Invalid Strands temperature_by_kind.");
        for (const auto& kind : {"noul", "choice", "score"}) {
            const auto& value = by_kind.isMember(kind) ? by_kind[kind] : metadata["temperature"];
            if (!value.isNumeric() || !std::isfinite(value.asDouble()) || value.asDouble() <= 0)
                throw std::runtime_error("Invalid Strands temperature for " + std::string(kind) + ".");
            temperatures[kind] = value.asDouble();
        }
    }
};

inline uint32_t strands_codepoint(const std::string& text, size_t& pos) {
    const auto first = static_cast<unsigned char>(text.at(pos++));
    if (first < 0x80) return first;
    unsigned extra = first >= 0xf0 ? 3 : first >= 0xe0 ? 2 : 1;
    uint32_t cp = first & (0x7f >> (extra + 1));
    while (extra--) {
        if (pos == text.size() || (static_cast<unsigned char>(text[pos]) & 0xc0) != 0x80)
            throw std::runtime_error("Invalid UTF-8 in Strands text or tokenizer.");
        cp = (cp << 6) | (static_cast<unsigned char>(text[pos++]) & 0x3f);
    }
    return cp;
}

inline bool strands_space(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 ||
        cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

inline std::string strands_whitespace(const std::string& text, bool collapse) {
    std::string result;
    size_t last_nonspace = 0;
    bool pending = false;
    for (size_t pos = 0; pos < text.size();) {
        const auto start = pos;
        if (strands_space(strands_codepoint(text, pos))) {
            if (!result.empty()) {
                if (!collapse) result.append(text, start, pos - start);
                pending = true;
            }
        } else {
            if (collapse && pending) result += ' ';
            result.append(text, start, pos - start);
            last_nonspace = result.size();
            pending = false;
        }
    }
    result.resize(last_nonspace);
    return result;
}

inline std::string strands_json(const Json::Value& value, size_t depth = 0) {
    if (value.isString()) {
        std::string result = "\"";
        constexpr char hex[] = "0123456789abcdef";
        for (unsigned char c : value.asString()) {
            switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20) {
                    result += "\\u00";
                    result += hex[c >> 4];
                    result += hex[c & 15];
                } else result += static_cast<char>(c);
            }
        }
        return result + '"';
    }
    if (value.type() == Json::realValue) {
        // Python json.dumps uses the shortest round-trip float, with fixed
        // notation for exponents -4..15. JsonCpp's older writer differs here.
        char buffer[64];
        if (!std::isfinite(value.asDouble())) throw std::runtime_error("Non-finite Strands JSON number.");
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value.asDouble(), std::chars_format::scientific);
        if (converted.ec != std::errc{}) throw std::runtime_error("Could not render Strands JSON number.");
        const std::string scientific(buffer, converted.ptr);
        const auto e = scientific.find('e');
        const int exponent = std::stoi(scientific.substr(e + 1));
        if (exponent < -4 || exponent >= 16) return scientific;
        const bool negative = scientific[0] == '-';
        auto digits = scientific.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
        digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
        const int point = exponent + 1;
        std::string fixed;
        if (point <= 0) fixed = "0." + std::string(-point, '0') + digits;
        else if (static_cast<size_t>(point) >= digits.size())
            fixed = digits + std::string(point - digits.size(), '0') + ".0";
        else fixed = digits.substr(0, point) + "." + digits.substr(point);
        return (negative ? "-" : "") + fixed;
    }
    if (!value.isObject() && !value.isArray()) return json_string(value);
    const bool object = value.isObject();
    const auto keys = object ? ordered_keys(value) : std::vector<std::string>{};
    std::string result = object ? "{" : "[";
    for (Json::ArrayIndex i = 0; i < value.size(); ++i) {
        result += i ? ",\n" : "\n";
        result += std::string((depth + 1) * 2, ' ');
        if (object) result += strands_json(keys[i]) + ": ";
        result += strands_json(object ? value[keys[i]] : value[i], depth + 1);
    }
    if (!value.empty()) result += "\n" + std::string(depth * 2, ' ');
    return result + (object ? "}" : "]");
}

inline std::string strands_content(const Json::Value& value) {
    if (value.isNull()) return "";
    return value.isString() ? strands_whitespace(value.asString(), false) : strands_json(value);
}

struct StrandsPrompt {
    std::string state;
    std::string question;
    std::vector<std::pair<size_t, size_t>> option_spans;
};

inline StrandsPrompt strands_prompt(const Request& request, const Question& q) {
    StrandsPrompt prompt;
    prompt.state = "<state>\n" + strands_content(request.state) + "\n</state>\n";
    const std::string header = q.type == "noul" ? "Decide whether the statement is true of the state." :
        q.type == "choice" ? "Select exactly one option." :
        "Rate the state against the ordered levels below (lowest first).";
    prompt.question = "<question type=\"" + q.type + "\">\n" + header + "\n" +
        strands_content(q.instructions) + "\n<options>\n";
    for (size_t i = 0; i < q.keys.size(); ++i) {
        const auto& key = q.keys[i];
        std::string desc;
        if (q.type == "noul") {
            desc = q.criteria.isMember(key) ? strands_content(q.criteria[key]) :
                key == "false" ? "the statement does not hold for this state" : "the statement holds for this state";
        } else {
            desc = strands_content(q.type == "choice" ? q.criteria[key] : q.criteria[Json::ArrayIndex(i)]);
        }
        desc = strands_whitespace(desc, true);
        const size_t start = prompt.question.size();
        prompt.question += std::to_string(i + 1) + ". " + key;
        if (!desc.empty()) prompt.question += " \xE2\x80\x94 " + desc;
        prompt.option_spans.emplace_back(start, prompt.question.size());
        prompt.question += '\n';
    }
    prompt.question += "</options>\n</question>\n<answer>";
    return prompt;
}

// tokenizers-cpp exposes IDs but not offsets. Recover ByteLevel offsets from the
// vocabulary, retaining NFC normalization and tokens which straddle a line end.
class StrandsTokenizer {
    tokenizers::Tokenizer& tokenizer_;
    std::map<int32_t, std::string> added_;
    std::map<uint32_t, unsigned char> bytes_;

public:
    static void validate_config(const Json::Value& config) {
        if (config["model"]["type"] != "BPE" || config["decoder"]["type"] != "ByteLevel")
            throw std::runtime_error("Strands requires a ByteLevel BPE tokenizer.");
        if ((!config["normalizer"].isNull() && config["normalizer"]["type"] != "NFC") ||
            (!config["post_processor"].isNull() &&
                (config["post_processor"]["type"] != "ByteLevel" || config["post_processor"]["trim_offsets"].asBool())))
            throw std::runtime_error("Strands requires NFC/identity normalization and no added post-processing tokens or trimmed offsets.");
    }

    StrandsTokenizer(tokenizers::Tokenizer& tokenizer, const Json::Value& config) : tokenizer_(tokenizer) {
        validate_config(config);
        unsigned next = 256;
        for (unsigned b = 0; b < 256; ++b) {
            const bool literal = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            bytes_[literal ? b : next++] = static_cast<unsigned char>(b);
        }
        for (const auto& token : config["added_tokens"]) {
            if (!token["id"].isInt() || !token["content"].isString() ||
                token["lstrip"].asBool() || token["rstrip"].asBool())
                throw std::runtime_error("Unsupported Strands added-token configuration.");
            added_[token["id"].asInt()] = token["content"].asString();
        }
    }

    std::string token_text(int32_t id) const {
        const auto found = added_.find(id);
        if (found != added_.end()) return found->second;
        const auto token = tokenizer_.IdToToken(id);
        if (token.empty()) throw std::runtime_error("Strands tokenizer returned an unknown token.");
        std::string text;
        for (size_t pos = 0; pos < token.size();) {
            const auto byte = bytes_.find(strands_codepoint(token, pos));
            if (byte == bytes_.end()) throw std::runtime_error("Invalid Strands ByteLevel vocabulary.");
            text += static_cast<char>(byte->second);
        }
        return text;
    }

    std::string normalized(const std::string& text) const {
        std::string result;
        for (auto id : tokenizer_.Encode(text)) result += token_text(id);
        return result;
    }

    std::vector<int64_t> encode(const std::string& text) const {
        const auto ids = tokenizer_.Encode(text);
        return {ids.begin(), ids.end()};
    }
};

struct StrandsInput {
    std::vector<int64_t> ids;
    std::vector<int64_t> option_positions;
    int64_t answer_position;
};

inline StrandsInput strands_input(const Request& request, const Question& q,
                                 const StrandsTokenizer& tokenizer, size_t context) {
    if (q.type == "score" && q.keys.size() < 2)
        throw ValidationError({"questions", q.id, "criteria"}, "Strands score requires 2 to 10 ordered levels.");
    auto prompt = strands_prompt(request, q);
    StrandsInput input;
    input.ids = tokenizer.encode(prompt.state);
    const auto question_ids = tokenizer.encode(prompt.question);
    if (input.ids.size() + question_ids.size() > context)
        throw ValidationError({"questions", q.id}, "Strands prompt exceeds " + std::to_string(context) + " tokens.");
    std::vector<std::pair<size_t, size_t>> offsets;
    size_t cursor = 0;
    for (auto id : question_ids) {
        const auto start = cursor;
        cursor += tokenizer.token_text(static_cast<int32_t>(id)).size();
        offsets.emplace_back(start, cursor);
    }
    for (const auto& span : prompt.option_spans) {
        const size_t start = tokenizer.normalized(prompt.question.substr(0, span.first)).size();
        const size_t end = tokenizer.normalized(prompt.question.substr(0, span.second)).size();
        int64_t last = -1;
        for (size_t i = 0; i < offsets.size(); ++i)
            if (offsets[i].second > offsets[i].first && offsets[i].first >= start && offsets[i].second <= end)
                last = static_cast<int64_t>(i);
        if (last < 0) throw std::runtime_error("Strands option span has no readout token.");
        input.option_positions.push_back(static_cast<int64_t>(input.ids.size()) + last);
    }
    input.ids.insert(input.ids.end(), question_ids.begin(), question_ids.end());
    input.answer_position = static_cast<int64_t>(input.ids.size()) - 1;
    return input;
}

} // namespace systemone
#endif
