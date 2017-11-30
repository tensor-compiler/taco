#ifndef SRC_UTIL_ENV_H_
#define SRC_UTIL_ENV_H_

#include <string>
#include <unistd.h>

#include "taco/error.h"

namespace taco {
namespace util {
std::string getFromEnv(std::string flag, std::string dflt);
std::string getTmpdir();

inline std::string getFromEnv(std::string flag, std::string dflt) {
  char const *ret = getenv(flag.c_str());
  if (!ret) {
    return dflt;
  } else {
    return std::string(ret);
  }
}

inline std::string getTmpdir() {
  // use POSIX logic for finding a temp dir
  auto tmpdirtemplate = getFromEnv("TMPDIR", "/tmp/");

  // if the directory does not have a trailing slash, add one
  if (tmpdirtemplate.back() != '/') {
    tmpdirtemplate += '/';
  }

  // ensure it is an absolute path
   taco_uassert(tmpdirtemplate.front() == '/') <<
    "The TMPDIR environment variable must be an absolute path";

  taco_uassert(access(tmpdirtemplate.c_str(), W_OK) == 0) <<
    "Unable to write to temporary directory for code generation. "
    "Please set the environment variable TMPDIR to somewhere writable";

  // ensure that we use a taco tmpdir unique to this process.

  tmpdirtemplate += "taco_tmp_XXXXXX";
  char tmpdir[tmpdirtemplate.length() + 1]
  std::strcpy(tmpdir, tmpdirtemplate.c_str());
  tmpdir = mkdtemp(tmpdir);
  taco_uassert(tmpdir != NULL) <<
    "Unable to create taco temporary directory for code generation. Please set"
    "the environment variable TMPDIR to somewhere searchable and writable";
  return std::string(tmpdir);
}

}}

#endif /* SRC_UTIL_ENV_H_ */
