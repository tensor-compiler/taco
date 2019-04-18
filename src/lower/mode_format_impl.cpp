#include "taco/lower/mode_format_impl.h"

#include <string>
#include <memory>
#include <vector>

#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

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
                               bool isCompact, bool hasCoordValIter, 
                               bool hasCoordPosIter, bool hasLocate, 
                               bool hasInsert, bool hasAppend) :
    name(name), isFull(isFull), isOrdered(isOrdered), isUnique(isUnique),
    isBranchless(isBranchless), isCompact(isCompact),
    hasCoordValIter(hasCoordValIter), hasCoordPosIter(hasCoordPosIter),
    hasLocate(hasLocate), hasInsert(hasInsert), hasAppend(hasAppend) {
}

ModeFormatImpl::~ModeFormatImpl() {
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

bool ModeFormatImpl::equals(const ModeFormatImpl& other) const {
  return (isFull == other.isFull &&
          isOrdered == other.isOrdered &&
          isUnique == other.isUnique &&
          isBranchless == other.isBranchless &&
          isCompact == other.isCompact);
}

bool operator==(const ModeFormatImpl& a, const ModeFormatImpl& b) {
  return (typeid(a) == typeid(b) && a.equals(b));
}

bool operator!=(const ModeFormatImpl& a, const ModeFormatImpl& b) {
  return !(a == b);
}

}
