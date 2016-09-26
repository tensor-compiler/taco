#ifndef TAC_FORMAT_H
#define TAC_FORMAT_H

#include <string>
#include <memory>
#include <ostream>

namespace tac {

class TreeLevel;

class Format {
public:
  Format() : forest{nullptr} {}

  Format(std::string format);

  friend std::ostream &operator<<(std::ostream&, const Format&);

private:
  std::shared_ptr<TreeLevel> forest;
};

}
#endif
