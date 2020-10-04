#include "taco/util/env.h"
#include <ftw.h>
#include <unistd.h>
#include <stdio.h>

namespace taco {
namespace util {

std::string cachedtmpdir = "";

static int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
    int rv = remove(fpath);
    taco_uassert(rv == 0) <<
      "Unable to create cleanup taco temporary directory. Sorry.";
    return rv;
}

void cachedtmpdirCleanup(void) {
  if (cachedtmpdir != ""){
    int rv = nftw(cachedtmpdir.c_str(), unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
    taco_uassert(rv == 0) <<
      "Unable to create cleanup taco temporary directory. Sorry.";
  }
}
}}
