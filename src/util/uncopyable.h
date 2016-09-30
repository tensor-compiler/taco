#ifndef TAC_UTIL_UNCOPYABLE_H
#define TAC_UTIL_UNCOPYABLE_H

namespace tac {
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
