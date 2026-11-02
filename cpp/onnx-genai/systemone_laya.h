#ifndef ONNX_SYSTEMONE_LAYA_H
#define ONNX_SYSTEMONE_LAYA_H

#include "systemone.h"
#include <array>
#include <charconv>
#include <functional>

namespace systemone {

inline std::string laya_json(const Json::Value& value, bool ascii = false) {
    // Match Python's shortest float representation and fixed/scientific thresholds.
    if (value.type() == Json::realValue) {
        const auto number = value.asDouble();
        if (!std::isfinite(number)) throw std::runtime_error("Laya JSON contains a non-finite number.");
        char buffer[128];
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), number, std::chars_format::scientific);
        if (converted.ec != std::errc{}) throw std::runtime_error("Could not serialize Laya JSON number.");
        std::string text(buffer, converted.ptr);
        const auto e = text.find('e');
        const int exponent = std::stoi(text.substr(e + 1));
        if (exponent < -4 || exponent >= 16) return text;
        const bool negative = text[0] == '-';
        auto digits = text.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
        digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
        std::string result = negative ? "-" : "";
        const int point = exponent + 1;
        if (point <= 0) return result + "0." + std::string(-point, '0') + digits;
        if (static_cast<size_t>(point) >= digits.size())
            return result + digits + std::string(point - digits.size(), '0') + ".0";
        return result + digits.substr(0, point) + "." + digits.substr(point);
    }
    if (value.isString() && !ascii) {
        // Some bundled JsonCpp libraries ignore emitUTF8; state must match ensure_ascii=False.
        std::string result = "\"";
        const char* hex = "0123456789abcdef";
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
        return result + "\"";
    }
    if (!value.isObject() && !value.isArray()) {
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        writer["emitUTF8"] = !ascii;
        auto text = Json::writeString(writer, value);
        if (ascii && value.isString()) {
            size_t pos = 0;
            while ((pos = text.find('\x7f', pos)) != std::string::npos) {
                text.replace(pos, 1, "\\u007f");
                pos += 6;
            }
        }
        return text;
    }
    const bool object = value.isObject();
    const auto keys = object ? ordered_keys(value) : std::vector<std::string>{};
    std::string result = object ? "{" : "[";
    for (Json::ArrayIndex i = 0; i < value.size(); ++i) {
        if (i) result += ", ";
        if (object) result += laya_json(keys[i], ascii) + ": ";
        result += laya_json(object ? value[keys[i]] : value[i], ascii);
    }
    return result + (object ? "}" : "]");
}

inline std::string laya_content(const Json::Value& value) {
    return value.isString() ? value.asString() : laya_json(value);
}

inline std::string laya_scrub(std::string text) {
    size_t pos = 0;
    while ((pos = text.find("[MASK]", pos)) != std::string::npos) text.replace(pos, 6, " ");
    return text;
}

struct LayaConfig {
    size_t max_len;
    size_t head_max_len;
    std::array<double, 3> temperature;
    Json::Value buckets;

    explicit LayaConfig(const Json::Value& config) {
        auto limit = [&](const char* key) {
            const auto& value = config[key];
            if (!value.isUInt64() || value.asUInt64() == 0 || value.asUInt64() > max_sequence_tokens)
                throw std::runtime_error(std::string("Invalid Laya limit: ") + key);
            return static_cast<size_t>(value.asUInt64());
        };
        if (!config.isObject()) throw std::runtime_error("Laya metadata must be an object.");
        max_len = limit("max_len");
        head_max_len = limit("head_max_len");
        if (head_max_len < 16 || head_max_len > max_len - std::min(max_len, size_t{4}))
            throw std::runtime_error("Laya head_max_len must be at least 16 and leave room for four special tokens.");
        auto checked = [](const Json::Value& value) {
            if (!value.isNumeric() || !std::isfinite(value.asDouble()) || value.asDouble() <= 0)
                throw std::runtime_error("Laya temperatures must be finite positive numbers.");
            return value.asDouble();
        };
        if (!config["temperature"].isArray() || config["temperature"].size() != 3)
            throw std::runtime_error("Laya temperature must contain choice, score and noul temperatures.");
        for (Json::ArrayIndex i = 0; i < 3; ++i) temperature[i] = checked(config["temperature"][i]);
        buckets = config["temperature_by_options"];
        if (!buckets.isNull() && !buckets.isObject())
            throw std::runtime_error("Laya temperature_by_options must be an object.");
        for (const auto& key : buckets.getMemberNames()) {
            const auto colon = key.find(':');
            const auto type = key.substr(0, colon);
            const auto size = colon == std::string::npos ? "" : key.substr(colon + 1);
            if ((type != "choice" && type != "score" && type != "noul") ||
                (size != "2" && size != "3-5" && size != "6-10" && size != "11+"))
                throw std::runtime_error("Unknown Laya temperature bucket: " + key);
            checked(buckets[key]);
        }
    }

    double for_question(const Question& q) const {
        const auto k = q.keys.size();
        const auto bucket = q.type + ":" + (k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+");
        return buckets.isMember(bucket) ? buckets[bucket].asDouble() :
            temperature[q.type == "choice" ? 0 : q.type == "score" ? 1 : 2];
    }
};

struct LayaInput {
    std::vector<int64_t> ids;
    std::vector<int64_t> markers;
    int64_t qtype;
};

inline LayaInput laya_sequence(const Request& request, const Question& q, const LayaConfig& config,
                              const std::array<int64_t, 4>& special,
                              const std::function<std::vector<int64_t>(const std::string&)>& encode) {
    const auto instruction = q.instructions.isString() ? q.instructions.asString() : laya_json(q.instructions, true);
    auto head = encode(q.type + " question: " + laya_scrub(instruction));
    std::vector<std::string> options;
    if (q.type == "choice") {
        for (const auto& key : q.keys)
            options.push_back(empty_content(q.criteria[key]) ? key : key + ": " + laya_content(q.criteria[key]));
    } else if (q.type == "score") {
        for (size_t i = 0; i < q.keys.size(); ++i)
            options.push_back("level " + std::to_string(i) + ": " + laya_content(q.criteria[static_cast<Json::ArrayIndex>(i)]));
    } else {
        options = {
            "false: " + (empty_content(q.criteria["false"]) ? "no, the statement does not hold" : laya_content(q.criteria["false"])),
            "true: " + (empty_content(q.criteria["true"]) ? "yes, the statement holds" : laya_content(q.criteria["true"]))
        };
    }
    std::vector<std::vector<int64_t>> encoded;
    size_t option_tokens = 0;
    for (const auto& option : options) {
        encoded.push_back(encode(" " + laya_scrub(option)));
        if (encoded.back().size() > 48)
            throw ValidationError({"questions", q.id, "criteria"}, "Laya option exceeds 48 tokens; shorten the option.");
        option_tokens += encoded.back().size() + 1;
    }
    if (option_tokens > config.head_max_len)
        throw ValidationError({"questions", q.id, "criteria"}, "Laya options exceed head_max_len.");
    const size_t budget = config.head_max_len - option_tokens;
    if (budget < 16) {
        const auto per = std::max(size_t{4}, (config.head_max_len - 16) / options.size());
        for (const auto& option : encoded)
            if (option.size() + 1 > per)
                throw ValidationError({"questions", q.id, "criteria"}, "Laya options require truncation; shorten the options.");
    }
    if (head.size() > std::max(size_t{8}, budget))
        throw ValidationError({"questions", q.id, "instructions"}, "Laya instructions exceed the head token budget.");
    LayaInput input{{special[0]}, {}, q.type == "choice" ? 0 : q.type == "score" ? 1 : 2};
    input.ids.insert(input.ids.end(), head.begin(), head.end());
    input.ids.push_back(special[1]);
    for (const auto& option : encoded) {
        input.markers.push_back(static_cast<int64_t>(input.ids.size()));
        input.ids.push_back(special[2]);
        input.ids.insert(input.ids.end(), option.begin(), option.end());
    }
    input.ids.push_back(special[1]);
    const auto state = encode(laya_scrub(laya_content(request.state)));
    if (input.ids.size() + state.size() + 1 > config.max_len)
        throw ValidationError({"questions", q.id}, "Laya state plus question exceeds max_len=" + std::to_string(config.max_len) + " tokens.");
    input.ids.insert(input.ids.end(), state.begin(), state.end());
    input.ids.push_back(special[1]);
    return input;
}

} // namespace systemone
#endif
