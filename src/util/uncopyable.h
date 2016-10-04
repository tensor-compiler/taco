#ifndef TACIT_UTIL_UNCOPYABLE_H
#define TACIT_UTIL_UNCOPYABLE_H

namespace tacit {
namespace util {

class Uncopyable {
protected:
  Uncopyable() = default;
  ~Uncopyable() = default;

private:
  Uncopyable(const Uncopyable&) = delete;
  Uncopyable& operator=(const Uncopyable&) = delete;
};

}}
#endif
