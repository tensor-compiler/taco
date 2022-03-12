#ifndef TACO_VERSION_H
#define TACO_VERSION_H

// This file contains version/config info, gathered at cmake config time.

#define TACO_BUILD_TYPE "Debug"
#define TACO_BUILD_DATE "2022-03-12"

#define TACO_BUILD_COMPILER_ID      "AppleClang"
#define TACO_BUILD_COMPILER_VERSION "11.0.3.11030032"

#define TACO_VERSION_MAJOR "0"
#define TACO_VERSION_MINOR "1"
// if taco starts using a patch number, add  here
// if taco starts using a tweak number, add  here

// For non-git builds, this will be an empty string.
#define TACO_VERSION_GIT_SHORTHASH "a9c46089"

#define TACO_FEATURE_OPENMP 0
#define TACO_FEATURE_PYTHON 0
#define TACO_FEATURE_CUDA   0

#endif /* TACO_VERSION_H */
