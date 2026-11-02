#include "../onnx-genai/systemone_onnx.h"
#include <iostream>

using namespace systemone;

static void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

template<class F> static void rejects(F operation, const std::string& context) {
    try { operation(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("Accepted invalid " + context);
}

template<class F> static void invalid_at(F operation, const std::string& field) {
    try { operation(); }
    catch (const ValidationError& error) {
        const auto location = parse_json(error.json())["detail"][0]["loc"];
        check(location[location.size() - 1].asString() == field, "Wrong validation location: " + error.json());
        return;
    }
    throw std::runtime_error("Accepted over-budget " + field);
}

static Json::Value metadata() {
    return parse_json(R"({"max_len":128,"head_max_len":64,"temperature":[2,3,4]})");
}

static Request sample() {
    return parse_request(R"({"model":"laya-latest","state":"state","questions":{
        "c":{"type":"choice","instructions":"choose","criteria":{"alpha":"one","beta":null}},
        "s":{"type":"score","instructions":"rate","criteria":["low","middle","high"]},
        "n":{"type":"noul","instructions":"true?"}
    }})");
}

static void pure_tests() {
    auto config = metadata();
    const auto request = sample();
    const LayaConfig defaults(config);
    check(defaults.for_question(request.questions[0]) == 2 &&
          defaults.for_question(request.questions[1]) == 3 &&
          defaults.for_question(request.questions[2]) == 4, "Missing buckets must use type temperatures");
    for (const auto& type : {"choice", "score", "noul"}) {
        for (const auto& bucket : {"2", "3-5", "6-10", "11+"})
            config["temperature_by_options"][std::string(type) + ":" + bucket] = 0.125;
    }
    LayaConfig buckets(config);
    for (const size_t size : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{6}, size_t{10}, size_t{11}, size_t{255}}) {
        for (auto q : request.questions) {
            q.keys.resize(size);
            check(buckets.for_question(q) == 0.125, "Wrong calibration bucket");
        }
    }
    auto sparse = metadata();
    sparse["temperature_by_options"]["choice:2"] = 0.00001;
    check(LayaConfig(sparse).for_question(request.questions[0]) == 0.00001, "Temperature was clamped");
    check(LayaConfig(sparse).for_question(request.questions[1]) == 3, "Missing bucket did not fall back");
    sparse["temperature_by_options"]["choice:2"] = 1e100;
    check(LayaConfig(sparse).for_question(request.questions[0]) == 1e100, "Large temperature was clamped");
    for (const auto& key : {"max_len", "head_max_len", "temperature"}) {
        auto bad = metadata(); bad.removeMember(key);
        rejects([&] { LayaConfig ignored(bad); }, std::string("missing ") + key);
    }
    for (const auto& key : {"max_len", "head_max_len"}) {
        for (const auto& value : {Json::Value(0), Json::Value(-1), Json::Value(8193), Json::Value(16.5),
                                  Json::Value("128"), Json::Value(true), Json::Value()}) {
            auto bad = metadata(); bad[key] = value;
            rejects([&] { LayaConfig ignored(bad); }, std::string("limit ") + key);
        }
    }
    for (const auto pair : {std::pair<int,int>{19,16}, {128,15}, {64,61}, {1,1}}) {
        auto bad = metadata(); bad["max_len"] = pair.first; bad["head_max_len"] = pair.second;
        rejects([&] { LayaConfig ignored(bad); }, "head relationship");
    }
    auto boundary = metadata(); boundary["max_len"] = 20; boundary["head_max_len"] = 16;
    check(LayaConfig(boundary).max_len == 20, "Rejected minimum valid limits");
    boundary["max_len"] = 8192; boundary["head_max_len"] = 8188;
    check(LayaConfig(boundary).max_len == 8192, "Rejected maximum valid limits");
    for (const auto& value : {Json::Value(0), Json::Value(-0.1), Json::Value("2"), Json::Value(true),
                              Json::Value(), Json::Value(std::numeric_limits<double>::infinity()),
                              Json::Value(std::numeric_limits<double>::quiet_NaN())}) {
        for (Json::ArrayIndex i = 0; i < 3; ++i) {
            auto bad = metadata(); bad["temperature"][i] = value;
            rejects([&] { LayaConfig ignored(bad); }, "temperature");
        }
        auto bad = metadata(); bad["temperature_by_options"]["choice:2"] = value;
        rejects([&] { LayaConfig ignored(bad); }, "bucket temperature");
    }
    for (const auto& value : {Json::Value(Json::arrayValue), Json::Value(2), Json::Value("2")}) {
        auto bad = metadata(); bad["temperature_by_options"] = value;
        rejects([&] { LayaConfig ignored(bad); }, "bucket container");
    }
    for (const auto& value : {parse_json("[]"), parse_json("[1,2]"), parse_json("[1,2,3,4]"), parse_json("{}")}) {
        auto bad = metadata(); bad["temperature"] = value;
        rejects([&] { LayaConfig ignored(bad); }, "temperature shape");
    }
    for (const auto& key : {"choice", "unknown:2", "choice:1", "choice:3", "score:11", "noul:2:extra"}) {
        auto bad = metadata(); bad["temperature_by_options"][key] = 1;
        rejects([&] { LayaConfig ignored(bad); }, "unknown bucket");
    }
    for (const auto& value : {Json::Value(), parse_json("[]"), Json::Value(true)})
        rejects([&] { LayaConfig ignored(value); }, "metadata container");

    check(laya_scrub("a[MASK][MASK]b [mask]") == "a  b [mask]", "MASK scrubbing changed");

    const std::array<int64_t,4> special{101,102,103,104};
    std::vector<std::string> texts;
    auto encoder = [&](const std::string& text) {
        texts.push_back(text);
        return std::vector<int64_t>{static_cast<int64_t>(200 + texts.size())};
    };
    const auto input = laya_sequence(request, request.questions[0], defaults, special, encoder);
    check(texts == std::vector<std::string>{"choice question: choose", " alpha: one", " beta", "state"}, "Choice rendering changed");
    check(input.ids == std::vector<int64_t>{101,201,102,103,202,103,203,102,204,102}, "Wrong sequence layout");
    check(input.markers == std::vector<int64_t>{3,5} && input.qtype == 0, "Wrong choice marker/qtype");
    texts.clear();
    const auto score = laya_sequence(request, request.questions[1], defaults, special, encoder);
    check(texts[1] == " level 0: low" && texts[3] == " level 2: high" && score.qtype == 1, "Wrong score labels/qtype");
    texts.clear();
    const auto noul = laya_sequence(request, request.questions[2], defaults, special, encoder);
    check(texts[1] == " false: no, the statement does not hold" &&
          texts[2] == " true: yes, the statement holds" && noul.qtype == 2, "Wrong noul defaults/qtype");
    auto injection = request;
    injection.state = "state[MASK]";
    injection.questions[0].instructions = parse_json(R"({"z":"é[MASK]"})");
    injection.questions[0].criteria["alpha"] = parse_json(R"({"nested":"[MASK]"})");
    texts.clear();
    laya_sequence(injection, injection.questions[0], defaults, special, encoder);
    check(texts[0] == "choice question: {\"z\": \"\\u00e9 \"}" &&
          texts[1] == " alpha: {\"nested\": \" \"}" && texts.back() == "state ", "Structured scrub/render mismatch");

    // Controlled token counts make exact rejection boundaries independent of any tokenizer.
    size_t head = 2, option = 2, state = 1;
    auto counted = [&](const std::string& text) {
        return std::vector<int64_t>(text == "state" ? state :
            text.rfind("choice question:", 0) == 0 ? head : option, 7);
    };
    auto limits = metadata(); limits["max_len"] = 128; limits["head_max_len"] = 114;
    auto run = [&] { return laya_sequence(request, request.questions[0], LayaConfig(limits), special, counted); };
    option = 48; run();
    option = 49; invalid_at(run, "criteria");
    option = 8; limits["head_max_len"] = 16; invalid_at(run, "criteria");
    option = 3; head = 8; run(); // Eight-token minimum instruction allowance.
    head = 9; invalid_at(run, "instructions");
    limits["head_max_len"] = 32; option = 8; head = 2;
    invalid_at(run, "criteria"); // Per-option budget is eight including MASK, not nine.
    option = 7; head = 16; run();
    head = 17; invalid_at(run, "instructions");
    limits["head_max_len"] = 64; option = 2; head = 58; run();
    head = 59; invalid_at(run, "instructions");
    head = 2; option = 2;
    const auto empty_state_size = run().ids.size() - state;
    state = 128 - empty_state_size; check(run().ids.size() == 128, "Rejected exact max_len");
    ++state; invalid_at(run, "c");
    const auto structured = parse_json(R"({"z":["é",true],"a":null})");
    check(laya_json(structured) == "{\"z\": [\"é\", true], \"a\": null}", "JSON spacing/order/UTF8 changed: " + laya_json(structured));
    check(laya_json(structured, true) == "{\"z\": [\"\\u00e9\", true], \"a\": null}", "Instruction ASCII JSON changed");
    check(laya_content("literal") == "literal" && laya_content(structured) == laya_json(structured), "Content rendering changed");
    for (const auto& value : std::vector<std::pair<double, std::string>>{
             {0.1, "0.1"}, {1.0, "1.0"}, {-0.0, "-0.0"}, {1e-4, "0.0001"},
             {1e-5, "1e-05"}, {1e15, "1000000000000000.0"}, {1e16, "1e+16"}}) {
        check(laya_json(Json::Value(value.first)) == value.second, "Python float rendering changed: " + value.second);
        check(laya_json(Json::Value(value.first), true) == value.second, "ASCII Python float rendering changed: " + value.second);
    }
    check(laya_json(Json::Value("\x7f"), true) == "\"\\u007f\"", "ASCII DEL must be escaped");
    check(laya_json(Json::Value("\x7f")) == "\"\x7f\"", "UTF8 DEL must remain literal");
}

static void dispatch_fixture_tests(const std::filesystem::path& root) {
    for (const auto& backend : {std::string("kev"), std::string("d1")}) {
        const auto directory = root / ("dispatch-" + backend);
        Model model(directory / (backend == "kev" ? "model.onnx" : "decoder_model_merged.onnx"), 1);
        check(std::string(model.backend()) == backend, "Enum dispatch changed " + backend + " backend");
        auto request = sample(); request.model = backend + "-latest";
        const auto actual = model.evaluate(request);
        size_t tokens = 0;
        KevInput kev;
        if (backend == "kev") { kev = model.prepare_kev(request); tokens = kev.ids.size(); }
        for (size_t row = 0; row < request.questions.size(); ++row) {
            const auto& q = request.questions[row];
            std::vector<double> scores;
            if (backend == "kev") {
                for (const auto end : kev.option_ends[row])
                    scores.push_back(static_cast<float>(kev.ids[end - 1]) * 0.1f + 0.001f);
            } else {
                const auto input = model.prepare_d1(request, q);
                tokens += input.ids.size();
                float sum = 0;
                for (const auto id : input.ids) sum += static_cast<float>(id);
                const float scale = sum * 0.0001f + 0.07f;
                for (const auto& group : input.readout)
                    scores.push_back(static_cast<float>(*std::max_element(group.begin(), group.end())) * scale);
            }
            const auto expected = answer(q, softmax(scores));
            auto single = request; single.questions = {q};
            const auto isolated = model.evaluate(single)["answers"][q.id];
            const auto& result = actual["answers"][q.id];
            if (q.type == "noul") {
                check(std::abs(result["noul"].asDouble() - expected["noul"].asDouble()) < 1e-6,
                      backend + " numeric dispatch mismatch");
                check(std::abs(result["noul"].asDouble() - isolated["noul"].asDouble()) < 1e-6,
                      backend + " mixed noul isolation mismatch");
            } else {
                for (const auto& key : q.keys) {
                    check(std::abs(result["probabilities"][key].asDouble() -
                                   expected["probabilities"][key].asDouble()) < 1e-6,
                          backend + " numeric dispatch mismatch");
                    check(std::abs(result["probabilities"][key].asDouble() -
                                   isolated["probabilities"][key].asDouble()) < 1e-6,
                          backend + " mixed isolation mismatch");
                }
            }
        }
        check(actual["model"] == model.id() && actual["usage"]["input_tokens"].asUInt64() == tokens &&
              actual["usage"]["output_tokens"] == 0, backend + " usage/identity dispatch mismatch");
        request.model = model.id(); model.evaluate(request);
        for (const auto& wrong : {"kev-latest", "d1-latest", "laya-latest"}) {
            if (backend + "-latest" == wrong) continue;
            request.model = wrong;
            invalid_at([&] { model.evaluate(request); }, "model");
        }
    }
    std::cout << "Kev/d1 enum dispatch fixtures passed.\n";
}

static void fixture_tests(const std::filesystem::path& root) {
    Model model(root / "model.onnx", 1);
    check(std::string(model.backend()) == "laya", "Wrong fixture backend");
    auto request = sample();
    request.model = model.id();
    auto choice = model.prepare_laya(request, request.questions[0]);
    check(choice.ids == std::vector<int64_t>{0,5,8,9,10,1,2,14,9,16,2,15,1,13,1},
          "WordLevel tokenization/layout differs from fixture");
    check(choice.markers == std::vector<int64_t>{6,10}, "Fixture markers wrong");
    const auto response = model.evaluate(request);
    size_t total = 0;
    const LayaConfig calibration(parse_json(read_file(root / "laya_config.json")));
    for (const auto& q : request.questions) {
        const auto input = model.prepare_laya(request, q);
        total += input.ids.size();
        double sum = 0;
        for (const auto id : input.ids) sum += id;
        std::vector<double> logits;
        for (const auto pos : input.markers) {
            const float score = static_cast<float>(input.ids[pos + 1]) *
                static_cast<float>(input.qtype + 1) * 0.07f +
                static_cast<float>(pos) * 0.013f +
                static_cast<float>(sum) * 0.0001f * static_cast<float>(pos);
            logits.push_back(score / calibration.for_question(q));
        }
        const auto expected = answer(q, softmax(logits));
        auto single = request; single.questions = {q};
        const auto isolated = model.evaluate(single)["answers"][q.id];
        const auto& actual = response["answers"][q.id];
        if (q.type == "noul") {
            check(std::abs(actual["noul"].asDouble() - expected["noul"].asDouble()) < 1e-6, "Fixture noul numeric mismatch");
            check(std::abs(actual["noul"].asDouble() - isolated["noul"].asDouble()) < 1e-6, "Mixed noul isolation failed");
        } else {
            for (const auto& key : q.keys) {
                check(std::abs(actual["probabilities"][key].asDouble() - expected["probabilities"][key].asDouble()) < 1e-6, "Fixture numeric mismatch");
                check(std::abs(actual["probabilities"][key].asDouble() - isolated["probabilities"][key].asDouble()) < 1e-6, "Mixed question isolation failed");
            }
            check(std::abs(actual["confidence"].asDouble() - expected["confidence"].asDouble()) < 1e-6, "TypeSafe confidence changed");
        }
    }
    check(response["usage"]["input_tokens"].asUInt64() == total && response["usage"]["output_tokens"] == 0,
          "Usage must count unpadded tokens");
    for (const auto& name : {model.id(), std::string("laya-latest")}) {
        request.model = name; model.evaluate(request);
    }
    for (const auto& name : {"kev", "kev-latest", "d1", "d1-latest", "laya", "local", "wrong-model"}) {
        request.model = name;
        rejects([&] { model.evaluate(request); }, std::string("wrong slot alias ") + name);
    }
    request.model = model.id();
    auto singleton = request; singleton.questions.resize(1); singleton.questions[0].keys.resize(1);
    check(model.evaluate(singleton)["answers"]["c"]["probabilities"]["alpha"].asDouble() == 1, "Singleton probability changed");
    auto padded = request;
    padded.state = std::string();
    for (int i = 0; i < 600; ++i) padded.state = padded.state.asString() + "state ";
    padded.questions.assign(64, request.questions[0]);
    invalid_at([&] { model.evaluate(padded); }, "questions");
    for (const auto& entry : std::filesystem::directory_iterator(root / "reject")) {
        rejects([&] { Model invalid(entry.path() / "model.onnx", 1); }, entry.path().filename().string());
    }
    Model nested(root / "nested-specials" / "model.onnx", 1);
    Model fallback(root / "no-buckets" / "model.onnx", 1);
    check(std::string(nested.backend()) == "laya" && std::string(fallback.backend()) == "laya", "Valid metadata rejected");
    request.model = "laya-latest";
    const auto nested_response = nested.evaluate(request);
    check(nested_response["answers"] == response["answers"], "Nested special metadata changes results");
    const auto fallback_response = fallback.evaluate(request);
    check(std::abs(fallback_response["answers"]["n"]["noul"].asDouble() -
                   response["answers"]["n"]["noul"].asDouble()) > 1e-4, "Missing buckets ignored default calibration");
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (entry.path().filename().string().rfind("runtime-", 0) != 0) continue;
        Model bad(entry.path() / "model.onnx", 1);
        rejects([&] { bad.evaluate(request); }, entry.path().filename().string());
    }
    std::cout << "Dynamic Laya fixture checks passed.\n";
    dispatch_fixture_tests(root);
}

int main(int argc, char** argv) {
    bool failed = false;
    try {
        pure_tests();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        failed = true;
    }
    if (argc > 1) {
        try { fixture_tests(std::filesystem::u8path(argv[1])); }
        catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            failed = true;
        }
    }
    if (!failed) std::cout << "Laya tests passed.\n";
    return failed ? 1 : 0;
}
