//
//  main.cpp
//  onnx-genai
//
//  Created by miyako on 2025/09/03.
//

#include "onnx-genai.h"

static void usage(void)
{
    fprintf(stderr, "Usage:  onnx-genai -m model -p port -\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c model   : %s\n", 'm' , "model");
    fprintf(stderr, " -%c path    : %s\n", 'o' , "output (default=stdout)");
    fprintf(stderr, " %c          : %s\n", '-' , "use stdin for input");
    fprintf(stderr, " -%c port    : %s\n", 'p' , "port");
    exit(1);
}

#if WITH_NATIVE_UTF_DECODE
static void utf16_to_utf8(const uint8_t *u16data, size_t u16size, std::string& u8) {
    
#ifdef __APPLE__
    CFStringRef str = CFStringCreateWithCharacters(kCFAllocatorDefault, (const UniChar *)u16data, u16size);
    if(str){
        size_t size = CFStringGetMaximumSizeForEncoding(CFStringGetLength(str), kCFStringEncodingUTF8) + sizeof(uint8_t);
        std::vector<uint8_t> buf(size+1);
        CFIndex len = 0;
        CFStringGetBytes(str, CFRangeMake(0, CFStringGetLength(str)), kCFStringEncodingUTF8, 0, true, (UInt8 *)buf.data(), buf.size(), &len);
        u8 = (const char *)buf.data();
        CFRelease(str);
    }else{
        u8 = "";
    }
#else
    int len = WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)u16data, u16size, NULL, 0, NULL, NULL);
    if(len){
        std::vector<uint8_t> buf(len + 1);
        WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)u16data, u16size, (LPSTR)buf.data(), buf.size(), NULL, NULL);
        u8 = (const char *)buf.data();
    }else{
        u8 = "";
    }
#endif
}
#endif

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
#define ARGS (OPTARG_T)L"m:o:-hp:"
#else
#define ARGS "m:o:-hp:"
#endif

#ifdef _WIN32
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

const std::string u_fffe = "\xEF\xBF\xBE";

int main(int argc, OPTARG_T argv[]) {
    

    const char* model_path = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4";
    
    std::cout << "Loading model from " << model_path << "..." << std::endl;

    try {
        // 2. Load Model and Create Generator
        auto model = OgaModel::Create(model_path);
        auto tokenizer = OgaTokenizer::Create(*model);
        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

        // 3. Encode Prompt
        const char* prompt = "<|user|>Tell me a joke.<|end|><|assistant|>";
        auto input_sequences = OgaSequences::Create();
        tokenizer->Encode(prompt, *input_sequences);

        // 4. Set Generation Parameters
        auto params = OgaGeneratorParams::Create(*model);
        params->SetSearchOption("max_length", 200); // Limit output length

        // 5. Generate Token by Token
        auto generator = OgaGenerator::Create(*model, *params); // <--- Changed back to original signature

        std::cout << "Output: ";
        
        while (!generator->IsDone()) {
            generator->GetLogits();
            generator->GenerateNextToken();
            
            // Get the newly generated token
            const int32_t* seq_data = generator->GetSequenceData(0);
            size_t seq_len = generator->GetSequenceCount(0);
            int32_t new_token = seq_data[seq_len - 1];

            // Decode and print
            const char* token_str = tokenizer_stream->Decode(new_token);
            std::cout << token_str << std::flush;
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    const OPTARG_T input_path  = NULL;
    const OPTARG_T output_path = NULL;
    
    std::vector<unsigned char>pdf_data(0);

    int ch;
    std::string text;
    bool rawText = false;
    OPTARG_T password = NULL;
    
    while ((ch = getopt(argc, argv, ARGS)) != -1){
        switch (ch){
            case 'i':
                input_path  = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 'p':
                password = optarg;
                break;
            case '-':
            {
                std::vector<uint8_t> buf(BUFLEN);
                size_t n;
                
                while ((n = fread(buf.data(), 1, buf.size(), stdin)) > 0) {
                    pdf_data.insert(pdf_data.end(), buf.begin(), buf.begin() + n);
                }
            }
                break;
            case 'r':
                rawText = true;
                break;
            case 'h':
            default:
                usage();
                break;
        }
    }
        
    if((!pdf_data.size()) && (input_path != NULL)) {
        FILE *f = _fopen(input_path, _rb);
        if(f) {
            _fseek(f, 0, SEEK_END);
            size_t len = (size_t)_ftell(f);
            _fseek(f, 0, SEEK_SET);
            pdf_data.resize(len);
            fread(pdf_data.data(), 1, pdf_data.size(), f);
            fclose(f);
        }
    }
    
    if(!pdf_data.size()) {
        usage();
    }
        
    if(!output_path) {
        std::cout << text << std::endl;
    }else{
        FILE *f = _fopen(output_path, _wb);
        if(f) {
            fwrite(text.c_str(), 1, text.length(), f);
            fclose(f);
        }
    }

    end:
    
//    FPDF_DestroyLibrary();
    
    return 0;
}
