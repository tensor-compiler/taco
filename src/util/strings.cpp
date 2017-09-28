#include "taco/util/strings.h"

#include <iostream>

using namespace std;

namespace taco {
namespace util {

vector<string> split(const string &str, const string &delim, bool keepDelim) {
  vector<string> results;
  size_t prev = 0;
  size_t next = 0;

  while ((next = str.find(delim, prev)) != std::string::npos) {
    if (next - prev != 0) {
      std::string substr = ((keepDelim) ? delim : "")
                         + str.substr(prev, next-prev);
      results.push_back(substr);
    }
    prev = next + delim.size();
  }

  if (prev < str.size()) {
    string substr = ((keepDelim) ? delim : "") + str.substr(prev);
    results.push_back(substr);
  }

  return results;
}

std::string repeat(std::string text, size_t n) {
  string str;
  for (size_t i = 0; i < n; i++) {
    str += text;
  }
  return str;
}

string fill(string text, char fill, size_t n) {
  size_t numfills = n - (text.size()+2);
  size_t prefix = numfills/2;
  size_t suffix = numfills/2 + (numfills % 2);
  return string(prefix,fill) + " " + text + " " + string(suffix,fill);
}

}}
