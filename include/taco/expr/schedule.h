#ifndef TACO_SCHEDULE_H
#define TACO_SCHEDULE_H

#include <memory>
#include <vector>

namespace taco {

class IndexVar;
class IndexExpr;


class OperatorSplit {
public:
  OperatorSplit(const IndexExpr& expr, const IndexVar& old,
                const IndexVar& left, const IndexVar& right);

  const IndexExpr& getExpr() const;
  const IndexVar& getOld() const;
  const IndexVar& getLeft() const;
  const IndexVar& getRight() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};


/// A schedule controls code generation and determines how index expression
/// should be computed.
class Schedule {
public:
  Schedule();

  const std::vector<OperatorSplit>& getOperatorSplits(const IndexExpr& expr);

  void addOperatorSplit(const OperatorSplit& split);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

}
#endif
