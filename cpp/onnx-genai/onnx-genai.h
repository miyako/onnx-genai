#ifndef __ONNX_GENAI_H__
#define __ONNX_GENAI_H__

#include <sstream>
#include <iostream>
#include <string>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <random>
#include <functional>
#include <filesystem>
#include <vector>
#include <exception>
#include <ort_genai.h>
#include "httplib.h"
#ifdef WIN32
#include <windows.h>
extern "C" {
    // 1. 自前で strcat_s 相当の関数を実装する
    // (依存ライブラリが期待するシグネチャに合わせる)
    errno_t __cdecl my_shim_strcat_s(char* dest, size_t destsz, const char* src) {
        if (!dest || !src) return EINVAL;

        size_t dlen = strlen(dest);
        size_t slen = strlen(src);

        if (dlen + slen + 1 > destsz) return ERANGE; // バッファオーバーフロー防止

        memcpy(dest + dlen, src, slen + 1);
        return 0;
    }

    // 2. リンカーが探している "__imp_strcat_s" という変数を定義し、
    //    上記関数のアドレスを入れておく。
    //    これで依存ライブラリはこの変数を参照して関数を呼び出せるようになります。
    void* __imp_strcat_s = (void*)my_shim_strcat_s;
}
#endif
#include "json/json.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_extensions.h>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>
#include <omp.h>
#include <fstream>
#include "tokenizers_cpp.h"

#define BUFLEN 4096

#ifdef __GNUC__
#define _fopen fopen
#define _fseek fseek
#define _ftell ftell
#define _rb "rb"
#define _wb "wb"
#else
#define _fopen _wfopen
#define _fseek _fseeki64
#define _ftell _ftelli64
#define _rb L"rb"
#define _wb L"wb"
#endif

#ifdef __GNUC__
#define OPTARG_T char*
#include <getopt.h>
#else
#ifndef _WINGETOPT_H_
#define _WINGETOPT_H_
#define OPTARG_T wchar_t*
#define main wmain
#define NULL    0
#define EOF    (-1)
#define ERR(s, c)    if(opterr){\
char errbuf[2];\
errbuf[0] = c; errbuf[1] = '\n';\
fputws(argv[0], stderr);\
fputws(s, stderr);\
fputwc(c, stderr);}
#ifdef __cplusplus
extern "C" {
#endif
    extern int opterr;
    extern int optind;
    extern int optopt;
    extern OPTARG_T optarg;
    extern int getopt(int argc, OPTARG_T *argv, OPTARG_T opts);
#ifdef __cplusplus
}
#endif
#endif  /* _WINGETOPT_H_ */
#endif

#pragma mark -

static std::string run_embeddings(
                                  Ort::Session *session,
                                  std::string& input,
                                  std::vector<const char*>&  input_names_c_array,
                                  size_t num_input_nodes,
                                  std::vector<const char*>&   output_names_c_array,
                                  size_t num_output_nodes);
static std::string run_inference(
                                 OgaModel* model,
                                 OgaTokenizer* tokenizer,
                                 const std::string& modelName,
                                 const std::string& fingerprint,
                                 long long created,
                                 unsigned int max_tokens,
                                 unsigned int top_k,
                                 double top_p,
                                 double temperature,
                                 unsigned int n,
                                 std::string prompt
                                 );
static void run_inference_stream(
                                 OgaModel* model,
                                 OgaTokenizer* tokenizer,
                                 const std::string& modelName,
                                 const std::string& fingerprint,
                                 long long created,
                                 unsigned int max_tokens,
                                 unsigned int top_k,
                                 double top_p,
                                 double temperature,
                                 unsigned int n,
                                 std::string prompt,
                                 std::function<bool(const std::string&)> on_token_generated
                                 );
static std::string create_stream_chunk(int n,
                                       const std::string& id,
                                       const std::string& model,
                                       const std::string& content,
                                       const std::string& fingerprint,
                                       bool finish = false);
#endif  /* __ONNX_GENAI_H__ */
