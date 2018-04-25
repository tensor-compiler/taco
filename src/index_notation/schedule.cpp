#include "taco/index_notation/schedule.h"

#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {

// class OperatorSplit
struct OperatorSplit::Content {
  IndexExpr expr;
  IndexVar old;
  IndexVar left;
  IndexVar right;
};

OperatorSplit::OperatorSplit(IndexExpr expr, IndexVar old,
                             IndexVar left, IndexVar right)
    : content(new Content) {
  content->expr = expr;
  content->old = old;
  content->left = left;
  content->right = right;
}

IndexExpr OperatorSplit::getExpr() const {
  return content->expr;
}

IndexVar OperatorSplit::getOld() const {
  return content->old;
}

IndexVar OperatorSplit::getLeft() const {
  return content->left;
}

IndexVar OperatorSplit::getRight() const {
  return content->right;
}

std::ostream& operator<<(std::ostream& os, const OperatorSplit& split) {
  return os << split.getExpr() << ": " << split.getOld() << " -> "
            << "(" << split.getLeft() << ", " << split.getRight() << ")";
}


// class Schedule
struct Schedule::Content {
  map<IndexExpr, vector<OperatorSplit>> operatorSplits;
};

Schedule::Schedule() : content(new Content) {
}

std::vector<OperatorSplit> Schedule::getOperatorSplits() const {
  std::vector<OperatorSplit> operatorSplits;
  for (auto& splits : content->operatorSplits) {
    util::append(operatorSplits, splits.second);
  }
  return operatorSplits;
}

vector<OperatorSplit> Schedule::getOperatorSplits(IndexExpr expr) const {
  return content->operatorSplits.at(expr);
}

void Schedule::addOperatorSplit(OperatorSplit split) {
  if (!util::contains(content->operatorSplits, split.getExpr())) {
    content->operatorSplits.insert({split.getExpr(), vector<OperatorSplit>()});
  }
  content->operatorSplits.at(split.getExpr()).push_back(split);
}

void Schedule::clearOperatorSplits() {
  content->operatorSplits.clear();
}

std::ostream& operator<<(std::ostream& os, const Schedule& schedule) {
  auto operatorSplits = schedule.getOperatorSplits();
  if (operatorSplits.size() > 0) {
    os << "Operator Splits:" << endl << util::join(operatorSplits, "\n");
  }
  return os;
}

}
