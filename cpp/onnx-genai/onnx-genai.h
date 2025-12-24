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
#endif
#include "json/json.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_extensions.h>
#include <cmath>
#include <numeric>

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

#endif  /* __ONNX_GENAI_H__ */
