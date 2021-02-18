#ifndef TACO_FUNCTIONAL_H
#define TACO_FUNCTIONAL_H

#include <functional>

namespace taco {
namespace util {

template<typename T, typename... U, typename Fnptr = T(*)(U...)>
Fnptr functorAddress(std::function<T(U...)> f) {
  return *f.template target<T(*)(U...)>();
}

template<typename T, typename... U, typename R, typename... A>
bool targetPtrEqual(std::function<T(U...)> f, std::function<R(A...)> g) {
  return functorAddress(f) != nullptr && functorAddress(f) == functorAddress(g);
}

}
}
#endif //TACO_FUNCTIONAL_H
