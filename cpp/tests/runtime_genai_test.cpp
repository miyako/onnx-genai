#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "onnxruntime_c_api.h"
#include "ort_genai.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: runtime-genai-test /path/to/tiny-random-gpt2-fp32\n";
        return 1;
    }
    struct Shutdown {
        ~Shutdown() { OgaShutdown(); }
    } shutdown;
    try {
        auto model = OgaModel::Create(argv[1]);
        auto params = OgaGeneratorParams::Create(*model);
        params->SetSearchOption("max_length", 4);
        params->SetSearchOption("min_length", 4);
        auto generator = OgaGenerator::Create(*model, *params);
        const int32_t input_ids[] = {0};
        generator->AppendTokens(input_ids, 1);
        while (!generator->IsDone()) {
            generator->GenerateNextToken();
        }
        if (generator->GetSequenceCount(0) != 4) {
            throw std::runtime_error("Expected a four-token generated sequence");
        }
        std::cout << "GenAI generation passed with ONNX Runtime "
                  << OrtGetApiBase()->GetVersionString() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
