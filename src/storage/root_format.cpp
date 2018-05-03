#include "taco/storage/root_format.h"

using namespace taco::ir;

namespace taco {

RootFormat::RootFormat() : ModeFormat("root", true, true, true, false, true, 
                                      true, false, false, true, false) {}

ModeType RootFormat::copy(
    const std::vector<ModeType::Property>& properties) const {
  return ModeType(std::make_shared<RootFormat>());
}

std::tuple<Stmt,Expr,Expr> RootFormat::getCoordIter(const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), 0ll, 1ll);
}

std::tuple<Stmt,Expr,Expr> RootFormat::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), i.back(), true);
}

Stmt RootFormat::getInsertCoord(const Expr& p, const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Expr RootFormat::getSize(const Expr& szPrev, 
    const ModeType::Mode& mode) const {
  return 1ll;
}

Stmt RootFormat::getInsertInit(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt RootFormat::getInsertFinalize(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Expr RootFormat::getArray(size_t idx, const ModeType::Mode& mode) const {
  return Expr();
}

}
