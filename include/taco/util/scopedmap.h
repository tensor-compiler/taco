#ifndef TACO_UTIL_SCOPEDMAP_H
#define TACO_UTIL_SCOPEDMAP_H

#include <list>
#include <map>
#include <ostream>

#include "taco/error.h"

namespace taco {
namespace util {

template <typename Key, typename Value>
class ScopedMap {
public:
  ScopedMap() {
    scope();
  }

  ~ScopedMap() {
    unscope();
  }

  /// Add a level of scoping.
  void scope() {
    scopes.push_front(std::map<Key,Value>());
  }

  /// Remove a level of scoping.
  void unscope() {
    scopes.pop_front();
  }

  void insert(const std::pair<Key, Value>& value) {
    scopes.front().insert(value);
  }

  const Value& get(const Key& key) const {
    for (auto& scope : scopes) {
      if (scope.find(key) != scope.end()) {
        return scope.at(key);
      }
    }
    taco_ierror << "Not in scope";
    return scopes.front().begin()->second;  // silence warnings
  }

  bool contains(const Key& key) {
    for (auto& scope : scopes) {
      if (scope.find(key) != scope.end()) {
        return true;
      }
    }
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, ScopedMap<Key,Value> smap) {
    os << "ScopedMap:" << std::endl;
    return os;
  }

private:
  std::list<std::map<Key, Value>> scopes;
};

}}
#endif
