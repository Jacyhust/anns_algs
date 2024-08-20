#pragma once
#include <cstdio>
#if defined(__GNUC__)
#include <atomic>
#include <cstring>
#include <cfloat>
#include <math.h>
static int fopen_s(FILE** pFile, const char* path, const char* mode)
{
	if ((*pFile = fopen64(path, mode)) == NULL) return 0;
	else return 1;
}
#elif defined _MSC_VER
#else
#endif