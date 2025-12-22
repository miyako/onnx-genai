//
//  main.cpp
//  onnx-genai
//
//  Created by miyako on 2025/09/03.
//

#include "onnx-genai.h"

static void usage(void)
{
    fprintf(stderr, "Usage:  onnx-genai -m model -t template -o output\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c model    : %s\n", 'm' , "model");
    fprintf(stderr, " -%c template : %s\n", 't' , "prompt template");
    fprintf(stderr, " -%c length   : %s\n", 'l' , "max token length");
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
#define ARGS (OPTARG_T)L"m:t:o:l:k:p:r:-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:t:o:l:k:p:r:-h"
#define _atoi atoi
#define _atof atof
#endif

int main(int argc, OPTARG_T argv[]) {
    
    const OPTARG_T model_path  = NULL;      //-m
    const OPTARG_T prompt_template  = NULL; //-t
    unsigned int max_tokens = 1024;         //-l
    unsigned int top_k = 40;                //-k
    double top_p = 1;                       //-p
    double temperature = 0.7;               //-r
    int ch;
    while ((ch = getopt(argc, argv, ARGS)) != -1){
        switch (ch){
            case 'm':
                model_path  = optarg;
                break;
            case 't':
                prompt_template  = optarg;
                break;
            case 'l':
                max_tokens = _atoi(optarg);
                break;
            case 'k':
                top_k = _atoi(optarg);
                break;
            case 'p':
                top_p = _atof(optarg);
                break;
            case 'h':
            default:
                usage();
                break;
        }
    }
    
    if((strlen(model_path) == 0) || (strlen(prompt_template) == 0)) {
        usage();
    }
    
    //const char* prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>";
    
    std::cerr << "Loading model from " << model_path << "..." << std::endl;

    try {
        // 2. Load Model and Create Generator
        auto model = OgaModel::Create(model_path);
        auto tokenizer = OgaTokenizer::Create(*model);
        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

        // 3. Encode Prompt
        auto input_sequences = OgaSequences::Create();
        tokenizer->Encode(prompt_template, *input_sequences);

        // 4. Set Generation Parameters
        auto params = OgaGeneratorParams::Create(*model);
        params->SetSearchOption("max_length", max_tokens);
        params->SetSearchOption("top_k", top_k);
        params->SetSearchOption("top_p", top_p);
        params->SetSearchOption("temperature", temperature);
        
        // 5. Generate Token by Token
        auto generator = OgaGenerator::Create(*model, *params);

        std::cout << "Output: ";
        
        while (1) {
 
            generator->GetLogits();
            generator->GenerateNextToken();
//            GenerateToken()
            
            if(generator->IsDone()) break;
            
            const int32_t* seq_data = generator->GetSequenceData(0);
            size_t seq_len = generator->GetSequenceCount(0);
            int32_t new_token = seq_data[seq_len - 1];
            const char* token_str = tokenizer_stream->Decode(new_token);
//            GetLastToken()
                
            std::cout << token_str << std::flush;
//            PrintLastToken()
        }
        
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
      
    return 0;
}
