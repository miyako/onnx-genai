//
//  main.cpp
//  onnx-genai
//
//  Created by miyako on 2025/09/03.
//

#include "onnx-genai.h"

static void usage(void)
{
    fprintf(stderr, "Usage:  onnx-genai -m model -i input\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c path     : %s\n", 'm' , "model");
    //
    fprintf(stderr, " -%c path     : %s\n", 'i' , "input");
    fprintf(stderr, " %c           : %s\n", '-' , "use stdin for input");
    fprintf(stderr, " -%c path     : %s\n", 'o' , "output (default=stdout)");
    //
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
#define ARGS (OPTARG_T)L"m:i:o:-h"
#define _atoi _wtoi
#define _atof _wtof
#define _strlen wcslen
#else
#define ARGS "m:i:o:-h"
#define _atoi atoi
#define _atof atof
#define _strlen strlen
#endif

static void parse_request(
              std::string &json,
              std::string &prompt,
              unsigned int *max_tokens,
              unsigned int *top_k,
              double *top_p,
              double *temperature,
              unsigned int *n) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    Json::CharReader *reader = builder.newCharReader();
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    delete reader;
    
    if(parse)
    {
        if(root.isObject())
        {
            Json::Value messages_node = root["messages"];
            if(messages_node.isArray())
            {
                prompt = "";
                for(Json::Value::const_iterator it = messages_node.begin() ; it != messages_node.end() ; it++)
                {
                    if(it->isObject())
                    {
                        Json::Value defaultValue = "";
                        Json::Value role = it->get("role", defaultValue);
                        Json::Value content = it->get("content", defaultValue);
                        if ((role.isString()) && (content.isString()))
                        {
                            std::string role_str = role.asString();
                            std::string content_str = content.asString();
                            if ((role_str.length() != 0) && (content_str.length() != 0))
                            {
                                prompt += "<|";
                                prompt += role_str;
                                prompt += "|>";
                                prompt += content_str;
                                prompt += "<|end|>";
                            }
                        }
                    }
                }
                if(prompt.length() != 0) prompt += "<|assistant|>";
            }
            Json::Value temperature_node = root["temperature"];
            if(messages_node.isDouble())
            {
                *temperature = temperature_node.asDouble();
            }
            Json::Value top_p_node = root["top_p"];
            if(top_p_node.isDouble())
            {
                *top_p = top_p_node.asDouble();
            }
            Json::Value top_k_node = root["top_k"];
            if(top_k_node.isInt())
            {
                *top_k = top_k_node.asInt();
            }
            Json::Value n_node = root["n"];
            if(n_node.isInt())
            {
                *n = n_node.asInt();
            }
            Json::Value max_tokens_node = root["max_tokens"];
            if(max_tokens_node.isInt())
            {
                *max_tokens = max_tokens_node.asInt();
            }
        }
    }
}

int main(int argc, OPTARG_T argv[]) {
        
    OPTARG_T model_path  = NULL;      //-m
    OPTARG_T input_path  = NULL;      //-i
    OPTARG_T output_path = NULL;      //-o
    
    unsigned int max_tokens = 2048;
    unsigned int top_k = 50;
    double top_p = 0.9;
    double temperature = 0.7;
    unsigned int n = 1;
        
    std::vector<unsigned char>request_json(0);
    
    int ch;
    
    while ((ch = getopt(argc, argv, ARGS)) != -1){
        switch (ch){
            case 'm':
                model_path  = optarg;
                break;
            case 'i':
                input_path  = optarg;
                break;
            case 'o':
                output_path  = optarg;
                break;
            case '-':
            {
                std::vector<uint8_t> buf(BUFLEN);
                size_t s;
                while ((s = fread(buf.data(), 1, buf.size(), stdin)) > 0) {
                    request_json.insert(request_json.end(), buf.begin(), buf.begin() + s);
                }
            }
                break;
            case 'h':
            default:
                usage();
                break;
        }
    }
    
    if ((model_path == NULL) || (_strlen(model_path) == 0)) {
        usage();
    }
    
    if ((!request_json.size()) && (input_path != NULL)) {
        FILE *f = _fopen(input_path, _rb);
        if(f) {
            _fseek(f, 0, SEEK_END);
            size_t len = (size_t)_ftell(f);
            _fseek(f, 0, SEEK_SET);
            request_json.resize(len);
            fread(request_json.data(), 1, request_json.size(), f);
            fclose(f);
        }
    }
    
    if (request_json.size() == 0) {
        usage();
    }
    
    std::string request((const char *)request_json.data(), request_json.size());
    std::string prompt;
    
    parse_request(request,
                  prompt,
                  &max_tokens,
                  &top_k,
                  &top_p,
                  &temperature,
                  &n);
    
    std::string response;
    
    try {
        // 2. Load Model and Create Generator
        auto model = OgaModel::Create(model_path);
        auto tokenizer = OgaTokenizer::Create(*model);
        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

        // 3. Encode Prompt
        auto input_sequences = OgaSequences::Create();
        tokenizer->Encode(prompt.c_str(), *input_sequences);
        size_t input_token_count = input_sequences->SequenceCount(0);

        // 4. Set Generation Parameters
        auto params = OgaGeneratorParams::Create(*model);
        params->SetSearchOption("max_length", (double)(input_token_count + max_tokens));
        params->SetSearchOption("top_k", top_k);
        params->SetSearchOption("top_p", top_p);
        params->SetSearchOption("temperature", temperature);
        params->SetSearchOption("num_return_sequences", n);
                
        auto generator = OgaGenerator::Create(*model, *params);
        generator->AppendTokenSequences(*input_sequences);

        // 5. Start Generating
        while (1) {
            generator->GenerateNextToken();
            
            if(generator->IsDone()) break;
            
            const int32_t* seq_data = generator->GetSequenceData(0);
            size_t seq_len = generator->GetSequenceCount(0);
            int32_t new_token = seq_data[seq_len - 1];
            const char* token_str = tokenizer_stream->Decode(new_token);
            std::cout << token_str << std::flush;
        }
        
        std::cout << std::endl;
        
        

    } catch (const std::exception& e) {
        Json::Value rootNode(Json::objectValue);
        Json::Value errorNode(Json::objectValue);
        rootNode["error"] = errorNode;
        errorNode["message"] = e.what();
        errorNode["type"] = "invalid_request_error";
        errorNode["param"] = Json::nullValue;
        errorNode["code"] = Json::nullValue;
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        response = Json::writeString(writer, rootNode);
    }
    
    if(!output_path) {
        std::cout << response << std::endl;
    }else{
        FILE *f = _fopen(output_path, _wb);
        if(f) {
            fwrite(response.c_str(), 1, response.length(), f);
            fclose(f);
        }
    }
      
    return 0;
}
