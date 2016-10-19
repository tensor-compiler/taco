#ifndef TACO_VAR_H
#define TACO_VAR_H

#include <string>
#include "util/comparable.h"
#include <memory>

namespace taco {

class Var : util::Comparable<Var> {
public:
  enum Kind { Free, Reduction };

private:
  struct Content {
    Var::Kind   kind;
    std::string name;
  };

public:
  Var() : content(nullptr) {
  }

  Var(const std::string& name, Kind kind = Kind::Free) : content(new Content) {
    content->name = name;
    content->kind = kind;
  }
  
  Kind getKind() const {
    return content->kind;
  }

  std::string getName() const {
    return content->name;
  }

  bool defined() {
    return content != nullptr;
  }

  friend bool operator==(const Var& l, const Var& r) {
    return l.content == r.content;
  }

  friend bool operator<(const Var& l, const Var& r) {
    return l.content < r.content;
  }

private:
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream& os, const Var& var);

}
#endif
