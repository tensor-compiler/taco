#ifndef TACO_VAR_H
#define TACO_VAR_H

#include <string>
#include <memory>

#include "util/comparable.h"

namespace taco {

class Var : public util::Comparable<Var> {
public:
  enum Kind { Free, Sum };

private:
  struct Content {
    Var::Kind   kind;
    std::string name;
  };

public:
  Var(Kind kind = Kind::Free);
  Var(const std::string& name, Kind kind = Kind::Free);

  std::string getName() const {return content->name;}

  Kind getKind() const {return content->kind;}

  bool isFree() const {return content->kind == Free;}

  bool isReduction() const {return content->kind != Free;}

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
