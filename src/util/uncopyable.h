#ifndef TACO_UTIL_UNCOPYABLE_H
#define TACO_UTIL_UNCOPYABLE_H

namespace taco {
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
