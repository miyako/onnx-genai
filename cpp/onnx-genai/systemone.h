#ifndef ONNX_SYSTEMONE_H
#define ONNX_SYSTEMONE_H

#include "json/json.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace systemone {

constexpr size_t max_body_bytes = 2 * 1024 * 1024;
constexpr size_t max_questions = 64;
constexpr size_t max_sequence_tokens = 8192;
constexpr size_t max_request_tokens = 32768;

inline std::string json_string(const Json::Value& value) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    writer["emitUTF8"] = true;
    return Json::writeString(writer, value);
}

class ValidationError : public std::runtime_error {
public:
    Json::Value detail;

    ValidationError(const std::vector<std::string>& path, const std::string& message,
                    const std::string& type = "value_error") : std::runtime_error(message) {
        Json::Value entry(Json::objectValue);
        entry["loc"] = Json::Value(Json::arrayValue);
        entry["loc"].append("body");
        for (const auto& part : path) entry["loc"].append(part);
        entry["msg"] = message;
        entry["type"] = type;
        detail = Json::Value(Json::arrayValue);
        detail.append(entry);
    }

    std::string json() const {
        Json::Value root(Json::objectValue);
        root["detail"] = detail;
        return json_string(root);
    }
};

inline Json::Value parse_json(const std::string& text) {
    Json::CharReaderBuilder builder;
    Json::CharReaderBuilder::strictMode(&builder.settings_);
    builder["stackLimit"] = 64;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Json::Value root;
    std::string errors;
    if (!reader->parse(text.data(), text.data() + text.size(), &root, &errors))
        throw ValidationError({}, "Invalid JSON: " + errors, "json_invalid");
    return root;
}

inline std::vector<std::string> ordered_keys(const Json::Value& object) {
    auto keys = object.getMemberNames();
    // JsonCpp stores objects in key order; prompts must retain the caller's option order.
    std::stable_sort(keys.begin(), keys.end(), [&](const std::string& a, const std::string& b) {
        return object[a].getOffsetStart() < object[b].getOffsetStart();
    });
    return keys;
}

inline bool is_content(const Json::Value& value) {
    return value.isString() || value.isObject() || value.isArray();
}

inline bool empty_content(const Json::Value& value) {
    return value.isNull() || (value.isString() && value.asString().empty()) ||
        ((value.isArray() || value.isObject()) && value.empty());
}

inline void check_fields(const Json::Value& object, const std::vector<std::string>& allowed,
                         const std::vector<std::string>& path) {
    for (const auto& key : object.getMemberNames()) {
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            auto field_path = path;
            field_path.push_back(key);
            throw ValidationError(field_path, "Unsupported field: " + key, "extra_forbidden");
        }
    }
}

struct Question {
    std::string id;
    std::string type;
    Json::Value instructions;
    Json::Value criteria;
    std::vector<std::string> keys;
};

struct Request {
    std::string model;
    Json::Value state;
    std::vector<Question> questions;
};

inline Request parse_request(const std::string& body) {
    if (body.size() > max_body_bytes)
        throw ValidationError({}, "Request exceeds the 2 MiB limit.");
    const auto root = parse_json(body);
    if (!root.isObject()) throw ValidationError({}, "Expected a JSON object.");
    check_fields(root, {"model", "state", "questions"}, {});
    if (!root["model"].isString() || root["model"].asString().empty())
        throw ValidationError({"model"}, "A non-empty model name is required.");
    if (!is_content(root["state"]))
        throw ValidationError({"state"}, "State must be a string, object or array.");
    const auto& questions = root["questions"];
    if (!questions.isObject() || questions.empty() || questions.size() > max_questions)
        throw ValidationError({"questions"}, "Provide 1 to 64 named questions.");

    Request request{root["model"].asString(), root["state"], {}};
    for (const auto& id : ordered_keys(questions)) {
        const auto& raw = questions[id];
        const std::vector<std::string> path{"questions", id};
        if (id.empty() || !raw.isObject())
            throw ValidationError(path, "Each question needs a non-empty name and an object value.");
        check_fields(raw, {"type", "instructions", "criteria"}, path);
        if (!raw["type"].isString())
            throw ValidationError({"questions", id, "type"}, "Question type is required.");
        Question q{id, raw["type"].asString(), raw["instructions"], raw["criteria"], {}};
        if (!q.instructions.isNull() && !is_content(q.instructions))
            throw ValidationError({"questions", id, "instructions"}, "Expected text, object, array or null.");
        if (q.type == "choice") {
            if (!q.criteria.isObject() || q.criteria.empty() || q.criteria.size() > 255)
                throw ValidationError({"questions", id, "criteria"}, "Choice requires 1 to 255 named options.");
            q.keys = ordered_keys(q.criteria);
            for (const auto& key : q.keys) {
                if (key.empty() || (!q.criteria[key].isNull() && !is_content(q.criteria[key])))
                    throw ValidationError({"questions", id, "criteria", key}, "Invalid option name or description.");
            }
        } else if (q.type == "score") {
            if (!q.criteria.isArray() || q.criteria.empty() || q.criteria.size() > 10)
                throw ValidationError({"questions", id, "criteria"}, "Score requires 1 to 10 ordered levels.");
            for (Json::ArrayIndex i = 0; i < q.criteria.size(); ++i) {
                if (!is_content(q.criteria[i]))
                    throw ValidationError({"questions", id, "criteria", std::to_string(i)}, "Invalid score level.");
                q.keys.push_back(std::to_string(i));
            }
        } else if (q.type == "noul") {
            if (!q.criteria.isNull() && !q.criteria.isObject())
                throw ValidationError({"questions", id, "criteria"}, "Noul criteria must be an object or null.");
            if (q.criteria.isObject()) {
                check_fields(q.criteria, {"true", "false"}, {"questions", id, "criteria"});
                for (const auto& key : q.criteria.getMemberNames())
                    if (!q.criteria[key].isNull() && !is_content(q.criteria[key]))
                        throw ValidationError({"questions", id, "criteria", key}, "Invalid yes/no description.");
            }
            if (empty_content(q.instructions) && empty_content(q.criteria["true"]) &&
                empty_content(q.criteria["false"]))
                throw ValidationError(path, "Noul requires instructions or a true/false description.");
            q.keys = {"false", "true"};
        } else {
            throw ValidationError({"questions", id, "type"}, "Supported question types are noul, choice and score.");
        }
        request.questions.push_back(std::move(q));
    }
    return request;
}

inline std::vector<double> softmax(const std::vector<double>& logits) {
    if (logits.empty()) throw std::runtime_error("Decision model returned no option scores.");
    const double peak = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities;
    double total = 0;
    for (double logit : logits) {
        if (!std::isfinite(logit)) throw std::runtime_error("Decision model returned a non-finite option score.");
        probabilities.push_back(std::exp(logit - peak));
        total += probabilities.back();
    }
    for (auto& p : probabilities) p /= total;
    return probabilities;
}

inline Json::Value answer(const Question& q, const std::vector<double>& p) {
    if (p.size() != q.keys.size()) throw std::runtime_error("Decision output does not match its options.");
    double total = 0;
    for (double value : p) {
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::runtime_error("Decision model returned an invalid probability.");
        total += value;
    }
    if (std::abs(total - 1) > 1e-6) throw std::runtime_error("Decision probabilities do not sum to one.");
    Json::Value result(Json::objectValue);
    result["type"] = q.type;
    if (q.type == "noul") {
        result["noul"] = p[1];
        return result;
    }
    const size_t mode = std::max_element(p.begin(), p.end()) - p.begin();
    result["probabilities"] = Json::Value(Json::objectValue);
    for (size_t i = 0; i < p.size(); ++i) result["probabilities"][q.keys[i]] = p[i];
    double confidence = 1;
    if (q.type == "choice") {
        result["choice"] = q.keys[mode];
        if (p.size() > 1) confidence = (p[mode] - 1.0 / p.size()) / (1 - 1.0 / p.size());
    } else {
        double score = 0, spread = 0, even_spread = 0;
        result["legend"] = Json::Value(Json::objectValue);
        for (size_t i = 0; i < p.size(); ++i) {
            score += i * p[i];
            spread += p[i] * std::abs(static_cast<double>(i) - mode);
            even_spread += std::abs(static_cast<double>(i) - (p.size() - 1) / 2.0) / p.size();
            result["legend"][q.keys[i]] = q.criteria[static_cast<Json::ArrayIndex>(i)];
        }
        result["score"] = score;
        if (p.size() > 1) confidence = 1 - spread / even_spread;
    }
    result["confidence"] = std::max(0.0, std::min(1.0, confidence));
    return result;
}

inline std::string ordered_json(const Json::Value& value, size_t depth = 0) {
    if (!value.isObject() && !value.isArray()) return json_string(value);
    const bool object = value.isObject();
    std::string result = object ? "{" : "[";
    const auto keys = object ? ordered_keys(value) : std::vector<std::string>{};
    for (Json::ArrayIndex i = 0; i < value.size(); ++i) {
        result += i ? ",\n" : "\n";
        result += std::string((depth + 1) * 2, ' ');
        if (object) result += json_string(keys[i]) + ": ";
        result += ordered_json(object ? value[keys[i]] : value[i], depth + 1);
    }
    if (!value.empty()) result += "\n" + std::string(depth * 2, ' ');
    return result + (object ? "}" : "]");
}

inline std::string kev_render(const Json::Value& value, size_t indent = 0) {
    if (value.isNull()) return "";
    if (value.isString()) return value.asString();
    if (value.isBool()) return value.asBool() ? "True" : "False";
    if (!value.isArray() && !value.isObject()) return json_string(value);
    std::string result;
    const std::string pad(indent * 2, ' ');
    if (value.isArray()) {
        for (const auto& item : value) {
            auto rendered = kev_render(item, indent + 1);
            const auto first = rendered.find_first_not_of(" \t\r\n");
            rendered = first == std::string::npos ? "" : rendered.substr(first);
            if (!result.empty()) result += "\n";
            result += pad + "- " + rendered;
        }
    } else {
        for (const auto& key : ordered_keys(value)) {
            if (!result.empty()) result += "\n";
            const auto& item = value[key];
            result += pad + key + (item.isObject() || item.isArray() ? ":\n" : ": ");
            result += kev_render(item, indent + 1);
        }
    }
    return result;
}

} // namespace systemone
#endif
