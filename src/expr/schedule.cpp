#include "taco/expr/schedule.h"

#include <map>

#include "taco/expr/expr.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {

// class OperatorSplit
struct OperatorSplit::Content {
  IndexExpr expr;
  IndexVar old;
  IndexVar left;
  IndexVar right;
};

OperatorSplit::OperatorSplit(const IndexExpr& expr, const IndexVar& old,
                             const IndexVar& left, const IndexVar& right)
    : content(new Content) {
  content->expr = expr;
  content->old = old;
  content->left = left;
  content->right = right;
}

const IndexExpr& OperatorSplit::getExpr() const {
  return content->expr;
}

const IndexVar& OperatorSplit::getOld() const {
  return content->old;
}

const IndexVar& OperatorSplit::getLeft() const {
  return content->left;
}

const IndexVar& OperatorSplit::getRight() const {
  return content->right;
}


// class Schedule
struct Schedule::Content {
  map<IndexExpr, vector<OperatorSplit>> operatorSplits;
};

Schedule::Schedule() : content(new Content) {
}

const vector<OperatorSplit>& Schedule::getOperatorSplits(const IndexExpr& expr){
  return content->operatorSplits.at(expr);
}

void Schedule::addOperatorSplit(const OperatorSplit& split) {
  if (!util::contains(content->operatorSplits, split.getExpr())) {
    content->operatorSplits.insert({split.getExpr(), vector<OperatorSplit>()});
  }
  content->operatorSplits.at(split.getExpr()).push_back(split);
}

}
