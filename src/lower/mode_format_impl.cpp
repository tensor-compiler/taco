#include "taco/lower/mode_format_impl.h"

#include <string>
#include <memory>
#include <vector>

#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

// class AttrQuery
struct AttrQuery::Content {
  std::vector<IndexVar> groupBy;
  std::vector<Attr> attrs;
};

AttrQuery::Attr::Attr(
    std::tuple<std::string,Aggregation,std::vector<IndexVar>> attr) :
        label(std::get<0>(attr)), aggr(std::get<1>(attr)), 
        params(std::get<2>(attr)) {
}

AttrQuery::AttrQuery(const std::vector<IndexVar>& groupBy, const Attr& attr) : 
    AttrQuery(groupBy, std::vector<Attr>{attr}) {
}

AttrQuery::AttrQuery(const std::vector<IndexVar>& groupBy,
                     const std::vector<Attr>& attrs) 
    : content(new Content) {
  taco_iassert(!attrs.empty());
  content->groupBy = groupBy;
  content->attrs = attrs;
}

AttrQuery::AttrQuery() : content(nullptr) {
}

const std::vector<IndexVar>& AttrQuery::getGroupBy() const {
  return content->groupBy;
}

const std::vector<AttrQuery::Attr>& AttrQuery::getAttrs() const {
  return content->attrs;
}

std::ostream& operator<<(std::ostream& os, const AttrQuery::Attr& attr) {
  switch (attr.aggr) {
    case AttrQuery::IDENTITY:
      os << "id";
      break;
    case AttrQuery::COUNT:
      os << "count";
      break;
    case AttrQuery::MIN:
      os << "min";
      break;
    case AttrQuery::MAX:
      os << "max";
      break;
    default:
      taco_iassert(false);
      break;
  }
  os << "(";
  if (attr.aggr != AttrQuery::IDENTITY) {
    os << util::join(attr.params);
  }
  os << ") as " << attr.label;
  return os;
}

std::ostream& operator<<(std::ostream&os, const AttrQuery& query) {
  os << "select [" << util::join(query.getGroupBy()) << "] -> "
     << util::join(query.getAttrs());
  return os;
}


// class AttrQueryResult
AttrQueryResult::AttrQueryResult(Expr resultVar, Expr resultValues) 
    : resultVar(resultVar), resultValues(resultValues) {}

Expr AttrQueryResult::getResult(const std::vector<Expr>& indices,
                                const std::string& attr) const {
  if (indices.empty()) {
    return resultValues;
  }

  Expr pos = 0;
  for (int i = 0; i < (int)indices.size(); ++i) {
    Expr dim = GetProperty::make(resultVar, TensorProperty::Dimension, i);
    pos = ir::Add::make(ir::Mul::make(pos, dim), indices[i]);
  }
  return Load::make(resultValues, pos);
}

std::ostream& operator<<(std::ostream& os, const AttrQueryResult& result) {
  return os << result.resultVar;
}


// class ModeFunction
struct ModeFunction::Content {
  Stmt body;
  vector<Expr> results;
};

ModeFunction::ModeFunction(Stmt body, const vector<Expr>& results)
    : content(new Content) {
  content->body = body;
  content->results = results;
}

ModeFunction::ModeFunction() : content(nullptr) {
}

ir::Stmt ModeFunction::compute() const {
  return content->body;
}

ir::Expr ModeFunction::operator[](size_t result) const {
  return content->results[result];
}

size_t ModeFunction::numResults() const {
  return content->results.size();
}

const std::vector<ir::Expr>& ModeFunction::getResults() const {
  return content->results;
}

bool ModeFunction::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const ModeFunction& modeFunction) {
  return os << modeFunction.compute() << endl
            << util::join(modeFunction.getResults());
}


// class ModeTypeImpl
ModeFormatImpl::ModeFormatImpl(const std::string name, bool isFull, 
                               bool isOrdered, bool isUnique, bool isBranchless, 
                               bool isCompact, bool isZeroless, bool isPadded,
                               bool hasCoordValIter, bool hasCoordPosIter, 
                               bool hasLocate, bool hasInsert, bool hasAppend, 
                               bool hasSeqInsertEdge, bool hasInsertCoord,
                               bool isYieldPosPure) :
    name(name), isFull(isFull), isOrdered(isOrdered), isUnique(isUnique),
    isBranchless(isBranchless), isCompact(isCompact), isZeroless(isZeroless),
    isPadded(isPadded), hasCoordValIter(hasCoordValIter),
    hasCoordPosIter(hasCoordPosIter), hasLocate(hasLocate),
    hasInsert(hasInsert), hasAppend(hasAppend), 
    hasSeqInsertEdge(hasSeqInsertEdge), hasInsertCoord(hasInsertCoord),
    isYieldPosPure(isYieldPosPure) {
}

ModeFormatImpl::~ModeFormatImpl() {
}

std::vector<AttrQuery> ModeFormatImpl::attrQueries(
    vector<IndexVar> parentCoords, vector<IndexVar> childCoords) const {
  return std::vector<AttrQuery>();
}
                                                  

ModeFunction ModeFormatImpl::coordIterBounds(vector<Expr> coords,
                                           Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeFormatImpl::coordIterAccess(ir::Expr parentPos,
                                           std::vector<ir::Expr> coords,
                                           Mode mode) const {
  return ModeFunction();
}


ModeFunction ModeFormatImpl::coordBounds(ir::Expr parentPos,
                                             Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeFormatImpl::posIterBounds(ir::Expr parentPos, Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeFormatImpl::posIterAccess(ir::Expr pos,
                                         std::vector<ir::Expr> coords,
                                         Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeFormatImpl::locate(ir::Expr parentPos,
                                  std::vector<ir::Expr> coords,
                                  Mode mode) const {
  return ModeFunction();
}
  
Stmt ModeFormatImpl::getInsertCoord(Expr p,
    const std::vector<Expr>& i, Mode mode) const {
  return Stmt();
}

Expr ModeFormatImpl::getWidth(Mode mode) const {
  return Expr();
}

Stmt ModeFormatImpl::getInsertInitCoords(Expr pBegin,
    Expr pEnd, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getInsertInitLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getInsertFinalizeLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getAppendCoord(Expr p, Expr i,
    Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getAppendEdges(Expr pPrev, Expr pBegin,
    Expr pEnd, Mode mode) const {
  return Stmt();
}

Expr ModeFormatImpl::getSize(Expr szPrev, Mode mode) const {
  return Expr();
}

Stmt ModeFormatImpl::getAppendInitEdges(Expr pPrevBegin,
    Expr pPrevEnd, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getAppendInitLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getAppendFinalizeLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Expr ModeFormatImpl::getAssembledSize(Expr prevSize, Mode mode) const {
  return Expr();
}

Stmt ModeFormatImpl::getSeqInitEdges(Expr prevSize, 
    std::vector<AttrQueryResult> queries, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getSeqInsertEdge(Expr parentPos, std::vector<Expr> coords,
    std::vector<AttrQueryResult> queries, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getInitCoords(Expr prevSize, 
    std::vector<AttrQueryResult> queries, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getInitYieldPos(Expr prevSize, Mode mode) const {
  return Stmt();
}

ModeFunction ModeFormatImpl::getYieldPos(Expr parentPos, 
    std::vector<Expr> coords, Mode mode) const {
  return ModeFunction();
}

Stmt ModeFormatImpl::getInsertCoord(Expr parentPos, Expr pos, 
    std::vector<Expr> coords, Mode mode) const {
  return Stmt();
}

Stmt ModeFormatImpl::getFinalizeYieldPos(Expr prevSize, Mode mode) const {
  return Stmt();
}

bool ModeFormatImpl::equals(const ModeFormatImpl& other) const {
  return (isFull == other.isFull &&
          isOrdered == other.isOrdered &&
          isUnique == other.isUnique &&
          isBranchless == other.isBranchless &&
          isCompact == other.isCompact &&
          isZeroless == other.isZeroless);
}

bool operator==(const ModeFormatImpl& a, const ModeFormatImpl& b) {
  return (typeid(a) == typeid(b) && a.equals(b));
}

bool operator!=(const ModeFormatImpl& a, const ModeFormatImpl& b) {
  return !(a == b);
}

}
