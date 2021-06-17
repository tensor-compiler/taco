#include "taco/index_notation/index_notation.h"

#include <algorithm>
#include <taco/ir/simplify.h>
#include "lower/mode_access.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"

#include "taco/index_notation/schedule.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_printer.h"
#include "taco/lower/lower.h"
#include "taco/lower/mode_format_impl.h"

#include "taco/util/name_generator.h"
#include "taco/util/scopedmap.h"

using namespace std;

namespace taco {

void IndexVarRel::print(std::ostream& stream) const {
  if (ptr == nullptr) {
    stream << "undefined";
  }
  else {
    switch(getRelType()) {
      case SPLIT:
        getNode<SplitRelNode>()->print(stream);
        break;
      case DIVIDE:
        getNode<DivideRelNode>()->print(stream);
        break;
      case POS:
        getNode<PosRelNode>()->print(stream);
        break;
      case FUSE:
        getNode<FuseRelNode>()->print(stream);
        break;
      case BOUND:
        getNode<BoundRelNode>()->print(stream);
        break;
      case PRECOMPUTE:
        getNode<PrecomputeRelNode>()->print(stream);
        break;
      default:
        taco_ierror;
    }
  }
}

bool IndexVarRel::equals(const IndexVarRel &rel) const {
  if (getRelType() != rel.getRelType()) {
    return false;
  }

  switch(getRelType()) {
    case SPLIT:
      return getNode<SplitRelNode>()->equals(*rel.getNode<SplitRelNode>());
    case DIVIDE:
      return getNode<DivideRelNode>()->equals(*rel.getNode<DivideRelNode>());
    case POS:
      return getNode<PosRelNode>()->equals(*rel.getNode<PosRelNode>());
    case FUSE:
      return getNode<FuseRelNode>()->equals(*rel.getNode<FuseRelNode>());
    case UNDEFINED:
      return true;
    case BOUND:
      return getNode<BoundRelNode>()->equals(*rel.getNode<BoundRelNode>());
    case PRECOMPUTE:
      return getNode<PrecomputeRelNode>()->equals(*rel.getNode<PrecomputeRelNode>());
    default:
      taco_ierror;
      return false;
  }
}

bool operator==(const IndexVarRel& a, const IndexVarRel& b) {
  return a.equals(b);
}

std::ostream& operator<<(std::ostream& stream, const IndexVarRel& rel) {
  rel.print(stream);
  return stream;
}

IndexVarRelType IndexVarRel::getRelType() const {
  if (ptr == NULL) return UNDEFINED;
  return getNode()->relType;
}

std::vector<ir::Expr> IndexVarRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_ierror;
  return {};
}

std::vector<ir::Expr> IndexVarRelNode::deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                        std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                       Iterators iterators, ProvenanceGraph provGraph) const {
  taco_ierror;
  return {};
}

ir::Expr IndexVarRelNode::recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const {
  taco_ierror;
  return {};
}

ir::Stmt IndexVarRelNode::recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_ierror;
  return {};
}

struct SplitRelNode::Content {
  IndexVar parentVar;
  IndexVar outerVar;
  IndexVar innerVar;
  size_t splitFactor;
};

SplitRelNode::SplitRelNode(IndexVar parentVar, IndexVar outerVar, IndexVar innerVar, size_t splitFactor)
  : IndexVarRelNode(SPLIT), content(new Content) {
  content->parentVar = parentVar;
  content->outerVar = outerVar;
  content->innerVar = innerVar;
  content->splitFactor = splitFactor;
}

const IndexVar& SplitRelNode::getParentVar() const {
  return content->parentVar;
}
const IndexVar& SplitRelNode::getOuterVar() const {
  return content->outerVar;
}
const IndexVar& SplitRelNode::getInnerVar() const {
  return content->innerVar;
}
const size_t& SplitRelNode::getSplitFactor() const {
  return content->splitFactor;
}

void SplitRelNode::print(std::ostream &stream) const {
  stream << "split(" << getParentVar() << ", " << getOuterVar() << ", " << getInnerVar() << ", " << getSplitFactor() << ")";
}

bool SplitRelNode::equals(const SplitRelNode &rel) const {
  return getParentVar() == rel.getParentVar() && getOuterVar() == rel.getOuterVar()
        && getInnerVar() == rel.getInnerVar() && getSplitFactor() == rel.getSplitFactor();
}

std::vector<IndexVar> SplitRelNode::getParents() const {
  return {getParentVar()};
}

std::vector<IndexVar> SplitRelNode::getChildren() const {
  return {getOuterVar(), getInnerVar()};
}

std::vector<IndexVar> SplitRelNode::getIrregulars() const {
  return {getOuterVar()};
}

std::vector<ir::Expr> SplitRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(computedBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentBound = computedBounds.at(getParentVar());
  bool outerVarDefined = definedVars.count(getOuterVar());
  bool innerVarDefined = definedVars.count(getInnerVar());

  if (provGraph.isPosVariable(getParentVar())) {
    return parentBound; // splitting pos space does not change coordinate bounds
  }

  ir::Expr splitFactorLiteral = ir::Literal::make(getSplitFactor(), variableExprs[getParentVar()].type());

  if (!outerVarDefined && !innerVarDefined) {
    return parentBound;
  }
  else if(outerVarDefined && !innerVarDefined) {
    // outerVar constrains space to a length splitFactor strip starting at outerVar * splitFactor
    ir::Expr minBound = parentBound[0];
    minBound = ir::Add::make(minBound, ir::Mul::make(variableExprs[getOuterVar()], splitFactorLiteral));
    ir::Expr maxBound = ir::Min::make(parentBound[1], ir::Add::make(minBound, splitFactorLiteral));
    return {minBound, maxBound};
  }
  else if(!outerVarDefined && innerVarDefined) {
    // when innerVar is defined first does not limit coordinate space
    return parentBound;
  }
  else {
    taco_iassert(outerVarDefined && innerVarDefined);
    // outerVar and innervar constrains space to a length 1 strip starting at outerVar * splitFactor + innerVar
    ir::Expr minBound = parentBound[0];
    minBound = ir::Add::make(minBound, ir::Add::make(ir::Mul::make(variableExprs[getOuterVar()], splitFactorLiteral), variableExprs[getInnerVar()]));
    ir::Expr maxBound = ir::Min::make(parentBound[1], ir::Add::make(minBound, ir::Literal::make(1, variableExprs[getParentVar()].type())));
    return {minBound, maxBound};
  }
}

std::vector<ir::Expr> SplitRelNode::deriveIterBounds(taco::IndexVar indexVar,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                     std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                     Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getOuterVar() || indexVar == getInnerVar());
  taco_iassert(parentIterBounds.size() == 1);
  taco_iassert(parentIterBounds.count(getParentVar()) == 1);

  std::vector<ir::Expr> parentBound = parentIterBounds.at(getParentVar());
  Datatype splitFactorType = parentBound[0].type();
  if (indexVar == getOuterVar()) {
    ir::Expr minBound = ir::Div::make(parentBound[0], ir::Literal::make(getSplitFactor(), splitFactorType));
    ir::Expr maxBound = ir::Div::make(ir::Add::make(parentBound[1], ir::Literal::make(getSplitFactor()-1, splitFactorType)), ir::Literal::make(getSplitFactor(), splitFactorType));
    return {minBound, maxBound};
  }
  else if (indexVar == getInnerVar()) {
    ir::Expr minBound = 0;
    ir::Expr maxBound = ir::Literal::make(getSplitFactor(), splitFactorType);
    return {minBound, maxBound};
  }
  taco_ierror;
  return {};
}

ir::Expr SplitRelNode::recoverVariable(taco::IndexVar indexVar,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                       Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getParentVar());
  taco_iassert(variableNames.count(getParentVar()) && variableNames.count(getOuterVar()) && variableNames.count(getInnerVar()));
  Datatype splitFactorType = variableNames[getParentVar()].type();
  return ir::Add::make(ir::Mul::make(variableNames[getOuterVar()], ir::Literal::make(getSplitFactor(), splitFactorType)), variableNames[getInnerVar()]);
}

ir::Stmt SplitRelNode::recoverChild(taco::IndexVar indexVar,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getOuterVar() || indexVar == getInnerVar());
  taco_iassert(variableNames.count(getParentVar()) && variableNames.count(getOuterVar()) && variableNames.count(getInnerVar()));
  Datatype splitFactorType = variableNames[getParentVar()].type();
  if (indexVar == getOuterVar()) {
    // outerVar = parentVar - innerVar
    ir::Expr subStmt = ir::Sub::make(variableNames[getParentVar()], variableNames[getInnerVar()]);
    if (emitVarDecl) {
      return ir::Stmt(ir::VarDecl::make(variableNames[getOuterVar()], subStmt));
    }
    else {
      return ir::Stmt(ir::Assign::make(variableNames[getOuterVar()], subStmt));
    }
  }
  else {
    // innerVar = parentVar - outerVar * splitFactor
    ir::Expr subStmt = ir::Sub::make(variableNames[getParentVar()],
                                     ir::Mul::make(variableNames[getOuterVar()], ir::Literal::make(getSplitFactor(), splitFactorType)));
    if (emitVarDecl) {
      return ir::Stmt(ir::VarDecl::make(variableNames[getInnerVar()], subStmt));
    }
    else {
      return ir::Stmt(ir::Assign::make(variableNames[getInnerVar()], subStmt));
    }
  }
}

bool operator==(const SplitRelNode& a, const SplitRelNode& b) {
  return a.equals(b);
}

struct DivideRelNode::Content {
  IndexVar parentVar;
  IndexVar outerVar;
  IndexVar innerVar;
  size_t divFactor;
};

DivideRelNode::DivideRelNode(IndexVar parentVar, IndexVar outerVar, IndexVar innerVar, size_t divFactor)
  : IndexVarRelNode(DIVIDE), content(new Content) {
  content->parentVar = parentVar;
  content->outerVar = outerVar;
  content->innerVar = innerVar;
  content->divFactor = divFactor;
}

const IndexVar& DivideRelNode::getParentVar() const {
  return content->parentVar;
}
const IndexVar& DivideRelNode::getOuterVar() const {
  return content->outerVar;
}
const IndexVar& DivideRelNode::getInnerVar() const {
  return content->innerVar;
}
const size_t& DivideRelNode::getDivFactor() const {
  return content->divFactor;
}

void DivideRelNode::print(std::ostream &stream) const {
  stream << "divide(" << getParentVar() << ", " << getOuterVar() << ", " << getInnerVar() << ", " << getDivFactor() << ")";
}

bool DivideRelNode::equals(const DivideRelNode &rel) const {
  return getParentVar() == rel.getParentVar() && getOuterVar() == rel.getOuterVar() &&
    getInnerVar() == rel.getInnerVar() && getDivFactor() == rel.getDivFactor();
}

std::vector<IndexVar> DivideRelNode::getParents() const {
  return {getParentVar()};
}

std::vector<IndexVar> DivideRelNode::getChildren() const {
  return {getOuterVar(), getInnerVar()};
}

std::vector<IndexVar> DivideRelNode::getIrregulars() const {
  return {getOuterVar()};
}

std::vector<ir::Expr> DivideRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(computedBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentBound = computedBounds.at(getParentVar());
  bool outerVarDefined = definedVars.count(getOuterVar());
  bool innerVarDefined = definedVars.count(getInnerVar());

  if (provGraph.isPosVariable(getParentVar()) || !outerVarDefined) {
    return parentBound; // splitting pos space does not change coordinate bounds
  }

  auto divFactorType = variableExprs[getParentVar()].type();
  auto divFactor = ir::Literal::make(getDivFactor(), divFactorType);
  auto divFactorMinusOne = ir::Literal::make(getDivFactor() - 1, divFactorType);
  auto dimLen = ir::Div::make(ir::Add::make(parentBound[1], divFactorMinusOne), divFactor);

  if(!innerVarDefined) {
    // outerVar constraints the space to a length parentBounds / divFactor strip starting at
    // outerVar * parentBounds / divFactor.
    auto lower = ir::Mul::make(variableExprs[getOuterVar()], dimLen);
    auto upper = ir::Mul::make(ir::Add::make(variableExprs[getOuterVar()], 1), dimLen);
    return {lower, upper};
  } else {
    taco_iassert(outerVarDefined && innerVarDefined);
    // outerVar and innerVar constrain space to a length 1 strip starting at
    // outerVar * parentBounds + innerVar.
    auto lower = ir::Add::make(ir::Mul::make(variableExprs[getOuterVar()], dimLen), variableExprs[getInnerVar()]);
    auto upper = ir::Min::make(parentBound[1], ir::Add::make(lower, 1));
    return {lower, upper};
  }
}

std::vector<ir::Expr> DivideRelNode::deriveIterBounds(taco::IndexVar indexVar,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                     std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                     Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getOuterVar() || indexVar == getInnerVar());
  taco_iassert(parentIterBounds.size() == 1);
  taco_iassert(parentIterBounds.count(getParentVar()) == 1);

  std::vector<ir::Expr> parentBound = parentIterBounds.at(getParentVar());
  Datatype divFactorType = parentBound[0].type();
  auto divFactor = ir::Literal::make(getDivFactor(), divFactorType);
  if (indexVar == getOuterVar()) {
    // The loop has been divided into divFactor pieces, so the outer variable
    // ranges from 0 to divFactor.
    ir::Expr minBound = 0;
    ir::Expr maxBound = divFactor;
    return {minBound, maxBound};
  }
  else if (indexVar == getInnerVar()) {
    // The inner loop ranges over a chunk of size parentBound / divFactor.
    ir::Expr minBound = ir::Div::make(parentBound[0], divFactor);
    ir::Expr maxBound = ir::Div::make(ir::Add::make(parentBound[1], ir::Literal::make(getDivFactor()-1, divFactorType)), divFactor);
    return {minBound, maxBound};
  }
  taco_ierror;
  return {};
}

ir::Expr DivideRelNode::recoverVariable(taco::IndexVar indexVar,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                       Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getParentVar());
  taco_iassert(variableNames.count(getParentVar()) && variableNames.count(getOuterVar()) && variableNames.count(getInnerVar()));
  // Extract divFactor and divFactor - 1.
  Datatype divFactorType = variableNames[getParentVar()].type();
  auto divFactor = ir::Literal::make(getDivFactor(), divFactorType);
  auto divFactorMinusOne = ir::Literal::make(getDivFactor() - 1, divFactorType);
  // Get the size of the dimension being iterated over.
  auto parentBounds = parentIterBounds.at(getParentVar());
  auto dimSize = ir::Sub::make(parentBounds[1], parentBounds[0]);
  // The bounds for the dimension are adjusted so that dimensions that aren't
  // divisible by divFactor have the last piece included.
  auto bounds = ir::Div::make(ir::Add::make(dimSize, divFactorMinusOne), divFactor);
  return ir::Add::make(ir::Mul::make(variableNames[getOuterVar()], bounds), variableNames[getInnerVar()]);
}

ir::Stmt DivideRelNode::recoverChild(taco::IndexVar indexVar,
                                    std::map<taco::IndexVar, taco::ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
  // We need bounds on the parent in order to recover the different
  // child values, but it doesn't seem like we have access to them here.
  taco_not_supported_yet;
  return ir::Stmt();
}

bool operator==(const DivideRelNode& a, const DivideRelNode& b) {
  return a.equals(b);
}

struct PosRelNode::Content {
  Content(IndexVar parentVar, IndexVar posVar, Access access) : parentVar(parentVar), posVar(posVar), access(access) {}
  IndexVar parentVar;
  IndexVar posVar;
  Access access;
};

PosRelNode::PosRelNode(IndexVar i, IndexVar ipos, const Access& access)
  : IndexVarRelNode(POS), content(new Content(i, ipos, access)) {
}

const IndexVar& PosRelNode::getParentVar() const {
  return content->parentVar;
}

const IndexVar& PosRelNode::getPosVar() const {
  return content->posVar;
}

const Access& PosRelNode::getAccess() const {
  return content->access;
}

void PosRelNode::print(std::ostream &stream) const {
  stream << "pos(" << getParentVar() << ", " << getPosVar() << ", " << getAccess() << ")";
}

bool PosRelNode::equals(const PosRelNode &rel) const {
  return getParentVar() == rel.getParentVar() && getPosVar() == rel.getPosVar()
         && getAccess() == rel.getAccess();
}

std::vector<IndexVar> PosRelNode::getParents() const {
  return {getParentVar()};
}

std::vector<IndexVar> PosRelNode::getChildren() const {
  return {getPosVar()};
}

std::vector<IndexVar> PosRelNode::getIrregulars() const {
  return {getPosVar()};
}

std::vector<ir::Expr> PosRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(computedBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentCoordBound = computedBounds.at(getParentVar());
  return parentCoordBound;
}

std::vector<ir::Expr> PosRelNode::deriveIterBounds(taco::IndexVar indexVar,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                     std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                     Iterators iterators,
                                                     ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getPosVar());
  taco_iassert(parentCoordBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentCoordBound = parentCoordBounds.at(getParentVar());

  if (provGraph.getUnderivedAncestors(indexVar).size() > 1) {
    // has fused take iterbounds instead
    // TODO: need to search segments by multiple underived for now just assume complete iteration
    Iterator accessIterator = getAccessIterator(iterators, provGraph);
    Iterator rootIterator = accessIterator;
    while(!rootIterator.isRoot()) {
      rootIterator = rootIterator.getParent();
    }
    ir::Expr parentSize = 1; // to find size of segment walk down sizes of iterator chain
    while (rootIterator != accessIterator) {
      rootIterator = rootIterator.getChild();
      if (rootIterator.hasAppend()) {
        parentSize = rootIterator.getSize(parentSize);
      } else if (rootIterator.hasInsert()) {
        parentSize = ir::Mul::make(parentSize, rootIterator.getWidth());
      }
    }
    return {ir::Literal::make(0), parentSize};
  }

  // locate position var for segment based on coordinate getParentVar()
  ir::Expr posVarExpr = variableNames[getPosVar()];
  return locateBounds(parentCoordBound, posVarExpr.type(), iterators, provGraph);
}

std::vector<ir::Expr> PosRelNode::locateBounds(std::vector<ir::Expr> coordBounds,
                                                   Datatype boundType,
                                                   Iterators iterators,
                                                   ProvenanceGraph provGraph) const {
  Iterator accessIterator = getAccessIterator(iterators, provGraph);
  ir::Expr parentPos = accessIterator.getParent().getPosVar();
  ModeFunction segment_bounds = accessIterator.posBounds(parentPos);
  vector<ir::Expr> binarySearchArgsStart = {
          getAccessCoordArray(iterators, provGraph),
          segment_bounds[0], // arrayStart
          segment_bounds[1], // arrayEnd
          coordBounds[0]
  };

  vector<ir::Expr> binarySearchArgsEnd = {
          getAccessCoordArray(iterators, provGraph),
          segment_bounds[0], // arrayStart
          segment_bounds[1], // arrayEnd
          coordBounds[1]
  };

  ir::Expr start = ir::Call::make("taco_binarySearchAfter", binarySearchArgsStart, boundType);
  // simplify start when this is 0
  ir::Expr simplifiedParentBound = ir::simplify(coordBounds[0]);
  if (isa<ir::Literal>(simplifiedParentBound) && to<ir::Literal>(simplifiedParentBound)->equalsScalar(0)) {
    start = segment_bounds[0];
  }
  ir::Expr end = ir::Call::make("taco_binarySearchAfter", binarySearchArgsEnd, boundType);
  // simplify end -> A1_pos[1] when parentBound[1] is max coord dimension
  simplifiedParentBound = ir::simplify(coordBounds[1]);
  if (isa<ir::GetProperty>(simplifiedParentBound) && to<ir::GetProperty>(simplifiedParentBound)->property == ir::TensorProperty::Dimension) {
    end = segment_bounds[1];
  }
  return {start, end};
}

Iterator PosRelNode::getAccessIterator(Iterators iterators, ProvenanceGraph provGraph) const {

  // when multiple underived ancestors, match iterator with max mode (iterate bottom level)
  vector<IndexVar> underivedParentAncestors = provGraph.getUnderivedAncestors(getParentVar());
  int max_mode = 0;
  for (IndexVar underivedParent : underivedParentAncestors) {
    size_t mode_index = 0; // which of the access index vars match?
    for (auto var : getAccess().getIndexVars()) {
      if (var == underivedParent) {
        break;
      }
      mode_index++;
    }
    taco_iassert(mode_index < getAccess().getIndexVars().size());
    int mode = getAccess().getTensorVar().getFormat().getModeOrdering()[mode_index];
    if (mode > max_mode) {
      max_mode = mode;
    }
  }

  // can't use default level iterator access function because mapping contents rather than pointer which is default to allow repeated operands
  std::map<ModeAccess, Iterator> levelIterators = iterators.levelIterators();
  ModeAccess modeAccess = ModeAccess(getAccess(), max_mode+1);
  for (auto levelIterator : levelIterators) {
    if (::taco::equals(levelIterator.first.getAccess(), modeAccess.getAccess()) && levelIterator.first.getModePos() == modeAccess.getModePos()) {
      return levelIterator.second;
    }
  }
  taco_ierror;
  return Iterator();
}

ir::Expr PosRelNode::getAccessCoordArray(Iterators iterators, ProvenanceGraph provGraph) const {
  return getAccessIterator(iterators, provGraph).getMode().getModePack().getArray(1);
}


ir::Expr PosRelNode::recoverVariable(taco::IndexVar indexVar,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                       Iterators iterators,
                                       std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                       std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                       ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getParentVar());
  taco_iassert(variableNames.count(getParentVar()) == 1 && variableNames.count(getPosVar()) == 1);
  taco_iassert(parentCoordBounds.count(getParentVar()) == 1);

  ir::Expr coord_array = getAccessCoordArray(iterators, provGraph);

  Iterator accessIterator = getAccessIterator(iterators, provGraph);
  ir::Expr parentPos = accessIterator.getParent().getPosVar();
  ModeFunction segment_bounds = accessIterator.posBounds(parentPos);

  // positions should be with respect to entire array not just segment so don't need to offset variable when projecting.
  ir::Expr project_result = ir::Load::make(coord_array, variableNames.at(getPosVar()));

  // but need to subtract parentvars start corodbound
  ir::Expr parent_value = ir::Sub::make(project_result, parentCoordBounds[getParentVar()][0]);

  return parent_value;
}

ir::Stmt PosRelNode::recoverChild(taco::IndexVar indexVar,
                                    std::map<taco::IndexVar, taco::ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getPosVar());
  taco_iassert(variableNames.count(getParentVar()) && variableNames.count(getPosVar()));
  // locate position var for segment based on coordinate getParentVar()
  ir::Expr posVarExpr = variableNames[getPosVar()];

  Iterator accessIterator = getAccessIterator(iterators, provGraph);
  ir::Expr parentPos = accessIterator.getParent().getPosVar();
  ModeFunction segment_bounds = accessIterator.posBounds(parentPos);
  vector<ir::Expr> binarySearchArgs = {
          getAccessCoordArray(iterators, provGraph),
          segment_bounds[0], // arrayStart
          segment_bounds[1], // arrayEnd
          variableNames[getParentVar()]
  };
  return ir::VarDecl::make(posVarExpr, ir::Call::make("taco_binarySearchAfter", binarySearchArgs, posVarExpr.type()));
}

bool operator==(const PosRelNode& a, const PosRelNode& b) {
  return a.equals(b);
}

struct FuseRelNode::Content {
  IndexVar outerParentVar;
  IndexVar innerParentVar;
  IndexVar fusedVar;
};

FuseRelNode::FuseRelNode(IndexVar outerParentVar, IndexVar innerParentVar, IndexVar fusedVar)
  : IndexVarRelNode(FUSE), content(new Content) {
  content->outerParentVar = outerParentVar;
  content->innerParentVar = innerParentVar;
  content->fusedVar = fusedVar;
}

const IndexVar& FuseRelNode::getOuterParentVar() const {
  return content->outerParentVar;
}
const IndexVar& FuseRelNode::getInnerParentVar() const {
  return content->innerParentVar;
}
const IndexVar& FuseRelNode::getFusedVar() const {
  return content->fusedVar;
}

void FuseRelNode::print(std::ostream &stream) const {
  stream << "fuse(" << getOuterParentVar() << ", " << getInnerParentVar() << ", " << getFusedVar() << ")";
}

bool FuseRelNode::equals(const FuseRelNode &rel) const {
  return getOuterParentVar() == rel.getOuterParentVar() && getInnerParentVar() == rel.getInnerParentVar()
         && getFusedVar() == rel.getFusedVar();
}

std::vector<IndexVar> FuseRelNode::getParents() const {
  return {getOuterParentVar(), getInnerParentVar()};
}

std::vector<IndexVar> FuseRelNode::getChildren() const {
  return {getFusedVar()};
}

std::vector<IndexVar> FuseRelNode::getIrregulars() const {
  return {getFusedVar()};
}

std::vector<ir::Expr> FuseRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(computedBounds.count(getOuterParentVar()) && computedBounds.count(getInnerParentVar()));
  return combineParentBounds(computedBounds[getOuterParentVar()], computedBounds[getInnerParentVar()]);
}

std::vector<ir::Expr> FuseRelNode::deriveIterBounds(taco::IndexVar indexVar,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                     std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                     Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getFusedVar());
  taco_iassert(parentIterBounds.count(getOuterParentVar()) && parentIterBounds.count(getInnerParentVar()));
  return combineParentBounds(parentIterBounds[getOuterParentVar()], parentIterBounds[getInnerParentVar()]);
}

ir::Expr FuseRelNode::recoverVariable(taco::IndexVar indexVar,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                       Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const {
  taco_iassert(variableNames.count(indexVar));
  taco_iassert(parentIterBounds.count(getInnerParentVar()));
  ir::Expr innerSize = ir::Sub::make(parentIterBounds[getInnerParentVar()][1], parentIterBounds[getInnerParentVar()][0]);

  if (indexVar == getOuterParentVar()) {
    // getOuterVar() = getFusedVar() / innerSize
    return ir::Div::make(variableNames[getFusedVar()], innerSize);
  }
  else if (indexVar == getInnerParentVar()) {
    if (provGraph.hasPosDescendant(getFusedVar()) && provGraph.isCoordVariable(getInnerParentVar())) {
      // getFusedVar() < innerSize (due to pos)
      return variableNames[getFusedVar()];
    }
    // getInnerVar() = getFusedVar() % innerSize
    return ir::Rem::make(variableNames[getFusedVar()], innerSize);
  }
  else {
    taco_unreachable;
    return ir::Expr();
  }
}

ir::Stmt FuseRelNode::recoverChild(taco::IndexVar indexVar,
                                    std::map<taco::IndexVar, taco::ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
//  taco_iassert(indexVar == fusedVar);
//  taco_iassert(variableNames.count(indexVar) && variableNames.count(outerParentVar) && variableNames.count(innerParentVar));
//  taco_iassert(parentCoordBounds.count(innerParentVar));
//  ir::Expr innerSize = ir::Sub::make(parentCoordBounds[innerParentVar][1], parentCoordBounds[innerParentVar][0]);
//  return ir::Add::make(ir::Mul::make(variableNames[outerParentVar], innerSize), variableNames[innerParentVar]);
  taco_not_supported_yet; // TODO: need to add parentIterBounds to recoverChild parameters
  return ir::Stmt();
}

// Combine two bounds
// if (i, j) where i in [a, b) and j in [c, d)
// then combined bound is [a * (d - c) + c, b * (d - c) + c)
// this results in (b - a) * (d - c) iterations while still being
// properly offset in cases where a != 0 or c != 0
std::vector<ir::Expr> FuseRelNode::combineParentBounds(std::vector<ir::Expr> outerParentBound, std::vector<ir::Expr> innerParentBound) const {
  ir::Expr innerSize = ir::Sub::make(innerParentBound[1], innerParentBound[0]);
  ir::Expr minBound = ir::Add::make(ir::Mul::make(outerParentBound[0], innerSize), innerParentBound[0]);
  ir::Expr maxBound = ir::Add::make(ir::Mul::make(outerParentBound[1], innerSize), innerParentBound[0]);
  return {minBound, maxBound};
}

bool operator==(const FuseRelNode& a, const FuseRelNode& b) {
  return a.equals(b);
}

// BoundRelNode
struct BoundRelNode::Content {
  IndexVar parentVar;
  IndexVar boundVar;
  size_t bound;
  BoundType boundType;
};

BoundRelNode::BoundRelNode(taco::IndexVar parentVar, taco::IndexVar boundVar, size_t bound,
                           taco::BoundType boundType) : IndexVarRelNode(BOUND), content(new Content) {
  content->parentVar = parentVar;
  content->boundVar = boundVar;
  content->bound = bound;
  content->boundType = boundType;
}

const IndexVar& BoundRelNode::getParentVar() const {
  return content->parentVar;
}
const IndexVar& BoundRelNode::getBoundVar() const {
  return content->boundVar;
}
const size_t& BoundRelNode::getBound() const {
  return content->bound;
}
const BoundType& BoundRelNode::getBoundType() const {
  return content->boundType;
}

void BoundRelNode::print(std::ostream &stream) const {
  stream << "bound(" << getParentVar() << ", " << getBoundVar() << ", " << getBound() << ", " << BoundType_NAMES[(int) getBoundType()] << ")";
}

bool BoundRelNode::equals(const BoundRelNode &rel) const {
  return getParentVar() == rel.getParentVar() &&
    getBoundVar() == rel.getBoundVar() && getBound() == rel.getBound() &&
    getBoundType() == rel.getBoundType();
}

std::vector<IndexVar> BoundRelNode::getParents() const {
  return {getParentVar()};
}

std::vector<IndexVar> BoundRelNode::getChildren() const {
  return {getBoundVar()};
}

std::vector<IndexVar> BoundRelNode::getIrregulars() const {
  return {getBoundVar()};
}

std::vector<ir::Expr> BoundRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  // coordinate bounds stay unchanged, only iteration bounds change
  taco_iassert(computedBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentCoordBound = computedBounds.at(getParentVar());
  return parentCoordBound;
}

std::vector<ir::Expr> BoundRelNode::deriveIterBounds(taco::IndexVar indexVar,
                                                   std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                   std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                   std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                   Iterators iterators,
                                                   ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getBoundVar());
  taco_iassert(parentCoordBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentCoordBound = parentCoordBounds.at(getParentVar());

  if (getBoundType() == BoundType::MaxExact) {
    return {parentCoordBound[0], ir::Literal::make(getBound(), parentCoordBound[1].type())};
  }
  else {
    taco_not_supported_yet;
  }
  return {};
}

ir::Expr BoundRelNode::recoverVariable(taco::IndexVar indexVar,
                                     std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                     Iterators iterators,
                                     std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                     ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getParentVar());
  taco_iassert(variableNames.count(getBoundVar()) == 1);
  return variableNames[getBoundVar()];
}

ir::Stmt BoundRelNode::recoverChild(taco::IndexVar indexVar,
                                  std::map<taco::IndexVar, taco::ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getBoundVar());
  taco_iassert(variableNames.count(getParentVar()) && variableNames.count(getBoundVar()));
  ir::Expr boundVarExpr = variableNames[getBoundVar()];
  return ir::VarDecl::make(boundVarExpr, variableNames[getParentVar()]);
}

bool operator==(const BoundRelNode& a, const BoundRelNode& b) {
  return a.equals(b);
}

// PrecomputeRelNode
struct PrecomputeRelNode::Content {
  IndexVar parentVar;
  IndexVar precomputeVar;
};

PrecomputeRelNode::PrecomputeRelNode(taco::IndexVar parentVar, taco::IndexVar precomputeVar)
  : IndexVarRelNode(PRECOMPUTE), content (new Content) {
  content->parentVar = parentVar;
  content->precomputeVar = precomputeVar;
}

const IndexVar& PrecomputeRelNode::getParentVar() const {
  return content->parentVar;
}

const IndexVar& PrecomputeRelNode::getPrecomputeVar() const {
  return content->precomputeVar;
}


void PrecomputeRelNode::print(std::ostream &stream) const {
  stream << "precompute(" << getParentVar() << ", " << getPrecomputeVar() << ")";
}

bool PrecomputeRelNode::equals(const PrecomputeRelNode &rel) const {
  return getParentVar() == rel.getParentVar() && getPrecomputeVar() == rel.getPrecomputeVar();
}

std::vector<IndexVar> PrecomputeRelNode::getParents() const {
  return {getParentVar()};
}

std::vector<IndexVar> PrecomputeRelNode::getChildren() const {
  return {getPrecomputeVar()};
}

std::vector<IndexVar> PrecomputeRelNode::getIrregulars() const {
  return {getPrecomputeVar()};
}

std::vector<ir::Expr> PrecomputeRelNode::computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(computedBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentCoordBound = computedBounds.at(getParentVar());
  return parentCoordBound;
}

std::vector<ir::Expr> PrecomputeRelNode::deriveIterBounds(taco::IndexVar indexVar,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                     std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                     Iterators iterators,
                                                     ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getPrecomputeVar());
  taco_iassert(parentIterBounds.count(getParentVar()) == 1);
  std::vector<ir::Expr> parentIterBound = parentIterBounds.at(getParentVar());
  return parentIterBound;
}

ir::Expr PrecomputeRelNode::recoverVariable(taco::IndexVar indexVar,
                                       std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                       Iterators iterators,
                                       std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                                       std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                       ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getParentVar());
  taco_iassert(variableNames.count(getPrecomputeVar()) == 1);
  return variableNames[getPrecomputeVar()];
}

ir::Stmt PrecomputeRelNode::recoverChild(taco::IndexVar indexVar,
                                    std::map<taco::IndexVar, taco::ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const {
  taco_iassert(indexVar == getPrecomputeVar());
  taco_iassert(variableNames.count(getParentVar()) && variableNames.count(getPrecomputeVar()));
  ir::Expr boundVarExpr = variableNames[getPrecomputeVar()];
  return ir::VarDecl::make(boundVarExpr, variableNames[getParentVar()]);
}

bool operator==(const PrecomputeRelNode& a, const PrecomputeRelNode& b) {
  return a.equals(b);
}

// class ProvenanceGraph
ProvenanceGraph::ProvenanceGraph(IndexStmt concreteStmt) {
  // Add all nodes (not all nodes may be scheduled)
  match(concreteStmt,
        std::function<void(const ForallNode*)>([&](const ForallNode* op) {
          nodes.insert(op->indexVar);
        })
  );

  // Get SuchThat node with relations
  if (!isa<SuchThat>(concreteStmt)) {
    // No relations defined
    return;
  }

  SuchThat suchThat = to<SuchThat>(concreteStmt);
  vector<IndexVarRel> relations = suchThat.getPredicate();

  for (IndexVarRel rel : relations) {
    std::vector<IndexVar> parents = rel.getNode()->getParents();
    std::vector<IndexVar> children = rel.getNode()->getChildren();
    for (IndexVar parent : parents) {
      nodes.insert(parent);
      childRelMap[parent] = rel;
      childrenMap[parent] = children;
    }

    for (IndexVar child : children) {
      nodes.insert(child);
      parentRelMap[child] = rel;
      parentsMap[child] = parents;
    }
  }
}

std::vector<IndexVar> ProvenanceGraph::getChildren(IndexVar indexVar) const {
  if (childrenMap.count(indexVar)) {
    return childrenMap.at(indexVar);
  }
  return {};
}

std::vector<IndexVar> ProvenanceGraph::getParents(IndexVar indexVar) const {
  if (parentsMap.count(indexVar)) {
    return parentsMap.at(indexVar);
  }
  return {};
}

std::vector<IndexVar> ProvenanceGraph::getFullyDerivedDescendants(IndexVar indexVar) const {
  // DFS to find all fully derived children
  std::vector<IndexVar> children = getChildren(indexVar);
  if (children.empty()) {
    return {indexVar};
  }

  std::vector<IndexVar> fullyDerivedChildren;
  for (IndexVar child : children) {
    std::vector<IndexVar> childFullyDerived = getFullyDerivedDescendants(child);
    fullyDerivedChildren.insert(fullyDerivedChildren.end(), childFullyDerived.begin(), childFullyDerived.end());
  }
  return fullyDerivedChildren;
}

std::vector<IndexVar> ProvenanceGraph::getUnderivedAncestors(IndexVar indexVar) const {
  // DFS to find all underived parents
  std::vector<IndexVar> parents = getParents(indexVar);
  if (parents.empty()) {
    return {indexVar};
  }

  std::vector<IndexVar> underivedParents;
  for (IndexVar parent : parents) {
    std::vector<IndexVar> parentUnderived = getUnderivedAncestors(parent);
    underivedParents.insert(underivedParents.end(), parentUnderived.begin(), parentUnderived.end());
  }
  return underivedParents;
}

bool ProvenanceGraph::getIrregularDescendant(IndexVar indexVar, IndexVar *irregularChild) const {
  if (isFullyDerived(indexVar) && isIrregular(indexVar)) {
    *irregularChild = indexVar;
    return true;
  }
  for (IndexVar child : getChildren(indexVar)) {
    if (getIrregularDescendant(child, irregularChild)) {
      return true;
    }
  }
  return false;
}

// A pos Iterator Descendant is first innermost variable that is pos
bool ProvenanceGraph::getPosIteratorAncestor(IndexVar indexVar, IndexVar *irregularChild) const {
  if (!isPosVariable(indexVar)) {
    return false;
  }

  if (isUnderived(indexVar)) {
    return false;
  }

  for (IndexVar parent : getParents(indexVar)) {
    if (isCoordVariable(parent)) {
      *irregularChild = indexVar;
      return true;
    }
    if (getPosIteratorAncestor(parent, irregularChild)) {
      return true;
    }
  }
  return false;
}

// A pos Iterator Descendant is first innermost variable that is pos
bool ProvenanceGraph::getPosIteratorDescendant(IndexVar indexVar, IndexVar *irregularChild) const {
  if (isPosVariable(indexVar)) {
    *irregularChild = indexVar;
    return true;
  }

  if (isFullyDerived(indexVar)) {
    return false;
  }

  if (childRelMap.at(indexVar).getRelType() == FUSE && isPosVariable(getChildren(indexVar)[0])) { // can't gain pos by fusing with pos variable
    return false;
  }

  if (getChildren(indexVar).size() == 1) {
    return getPosIteratorDescendant(getChildren(indexVar)[0], irregularChild);
  }
  for (IndexVar child : getChildren(indexVar)) {
    if (!util::contains(childRelMap.at(indexVar).getNode()->getIrregulars(), child)) { // is irregularity not maintained through relationship
      return getPosIteratorDescendant(child, irregularChild);
    }
  }
  return false;
}

bool ProvenanceGraph::getPosIteratorFullyDerivedDescendant(IndexVar indexVar, IndexVar *irregularChild) const {
  if (isFullyDerived(indexVar) || childRelMap.at(indexVar).getRelType() == PRECOMPUTE) {
    if (isPosVariable(indexVar)) {
      *irregularChild = indexVar;
      return true;
    }
    return false;
  }

  if (childRelMap.at(indexVar).getRelType() == FUSE && isPosVariable(indexVar)) { // can't iterate pos through fuse
    return false;
  }

  if (getChildren(indexVar).size() == 1) {
    return getPosIteratorFullyDerivedDescendant(getChildren(indexVar)[0], irregularChild);
  }
  for (IndexVar child : getChildren(indexVar)) {
    if (!util::contains(childRelMap.at(indexVar).getNode()->getIrregulars(), child)) { // is irregularity not maintained through relationship
      return getPosIteratorFullyDerivedDescendant(child, irregularChild); // TODO: need new classification rather than reusing irregular
    }
  }
  return false;
}

bool ProvenanceGraph::isIrregular(IndexVar indexVar) const {
  if (isUnderived(indexVar)) {
    return true;
  }

  IndexVarRel rel = parentRelMap.at(indexVar);
  std::vector<IndexVar> irregulars = rel.getNode()->getIrregulars();
  auto it = std::find (irregulars.begin(), irregulars.end(), indexVar);
  if (it == irregulars.end()) {
    // variable does not maintain irregular status through relationship
    return false;
  }

  for (const IndexVar& parent : getParents(indexVar)) {
    if (isIrregular(parent)) {
      return true;
    }
  }
  return false;
}

bool ProvenanceGraph::isUnderived(taco::IndexVar indexVar) const {
  return getParents(indexVar).empty();
}

bool ProvenanceGraph::isDerivedFrom(taco::IndexVar indexVar, taco::IndexVar ancestor) const {
  for (IndexVar parent : getParents(indexVar)) {
    if (parent == ancestor) {
      return true;
    }
    if(isDerivedFrom(parent, ancestor)) {
      return true;
    }
  }
  return false;
}

bool ProvenanceGraph::isFullyDerived(taco::IndexVar indexVar) const {
  return getChildren(indexVar).empty();
}

bool ProvenanceGraph::isAvailable(IndexVar indexVar, std::set<IndexVar> defined) const {
  for (const IndexVar& parent : getParents(indexVar)) {
    if (!defined.count(parent)) {
      return false;
    }
  }
  return true;
}

bool ProvenanceGraph::isRecoverable(taco::IndexVar indexVar, std::set<taco::IndexVar> defined) const {
  // all children are either defined or recoverable from their children
  // This checks the definedVars list to determine where in the statement the variables are trying to be
  // recovered from ( either on the producer or consumer side of a where stmt or not in a where stmt)
  vector<IndexVar> producers;
  vector<IndexVar> consumers;
  for (auto& def : defined) {
    if (childRelMap.count(def) && childRelMap.at(def).getRelType() == IndexVarRelType::PRECOMPUTE) {
      consumers.push_back(def);
    }
    if (parentRelMap.count(def) && parentRelMap.at(def).getRelType() == IndexVarRelType::PRECOMPUTE) {
      producers.push_back(def);
    }
  }

  return isRecoverablePrecompute(indexVar, defined, producers, consumers);
}

bool ProvenanceGraph::isRecoverablePrecompute(taco::IndexVar indexVar, std::set<taco::IndexVar> defined,
                                              vector<IndexVar> producers, vector<IndexVar> consumers) const {
  vector<IndexVar> childPrecompute;
  if (std::find(consumers.begin(), consumers.end(), indexVar) != consumers.end()) {
    return true;
  }
  if (!producers.empty() && (childRelMap.count(indexVar) &&
                             childRelMap.at(indexVar).getRelType() == IndexVarRelType::PRECOMPUTE)) {
    auto precomputeChild = getChildren(indexVar)[0];
    if (std::find(producers.begin(), producers.end(), precomputeChild) != producers.end()) {
      return true;
    }
    return isRecoverablePrecompute(precomputeChild, defined, producers, consumers);
  }
  for (const IndexVar& child : getChildren(indexVar)) {
    if (!defined.count(child) && (isFullyDerived(child) ||
                                  !isRecoverablePrecompute(child, defined, producers, consumers))) {
      return false;
    }
  }
  return true;
}

bool ProvenanceGraph::isChildRecoverable(taco::IndexVar indexVar, std::set<taco::IndexVar> defined) const {
  // at most 1 unknown in relation
  int count_unknown = 0;
  for (const IndexVar& parent : getParents(indexVar)) {
    if (!defined.count(parent)) {
      count_unknown++;
    }
    for (const IndexVar& sibling : getChildren(parent)) {
      if (!defined.count(sibling)) {
        count_unknown++;
      }
    }
  }
  return count_unknown <= 1;
}

// in terms of joined spaces
void ProvenanceGraph::addRelativeBoundsToMap(IndexVar indexVar, std::set<IndexVar> alreadyDefined, std::map<IndexVar, std::vector<ir::Expr>> &bounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators) const {
  // derive bounds of parents and use to construct bounds
  if (isUnderived(indexVar)) {
    taco_iassert(bounds.count(indexVar));
    return; // underived bound should already be in bounds
  }

  for (IndexVar parent : getParents(indexVar)) {
    addRelativeBoundsToMap(parent, alreadyDefined, bounds, variableExprs, iterators);
  }

  IndexVarRel rel = parentRelMap.at(indexVar);
  bounds[indexVar] = rel.getNode()->computeRelativeBound(alreadyDefined, bounds, variableExprs, iterators, *this);
}

void ProvenanceGraph::computeBoundsForUnderivedAncestors(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> relativeBounds, std::map<IndexVar, std::vector<ir::Expr>> &computedBounds) const {
  std::vector<IndexVar> underivedAncestors = getUnderivedAncestors(indexVar);
  // taco_iassert(underivedAncestors.size() == 1); // TODO: fuse

  computedBounds[underivedAncestors[0]] = relativeBounds[indexVar];
}

std::map<IndexVar, std::vector<ir::Expr>> ProvenanceGraph::deriveCoordBounds(std::vector<IndexVar> derivedVarOrder, std::map<IndexVar, std::vector<ir::Expr>> underivedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators) const {
  std::map<IndexVar, std::vector<ir::Expr>> computedCoordbounds = underivedBounds;
  std::set<IndexVar> defined;
  for (IndexVar indexVar : derivedVarOrder) {
    if (indexVar != derivedVarOrder.back()) {
      for (auto recoverable : newlyRecoverableParents(indexVar, defined)) {
        defined.insert(recoverable);
      }
      defined.insert(indexVar);
    }
    if (isUnderived(indexVar)) {
      continue; // underived indexvar can't constrain bounds
    }

    // add all relative coord bounds of nodes along derivation path to map.
    std::map<IndexVar, std::vector<ir::Expr>> relativeBounds = underivedBounds;
    addRelativeBoundsToMap(indexVar, defined, relativeBounds, variableExprs, iterators);

    // modify bounds for affected underived
    computeBoundsForUnderivedAncestors(indexVar, relativeBounds, computedCoordbounds);
  }
  return computedCoordbounds;
}

std::vector<ir::Expr> ProvenanceGraph::deriveIterBounds(IndexVar indexVar, std::vector<IndexVar> derivedVarOrder, std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                                                  std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const {
  // strategy is to start with underived variable bounds and propagate through each step on return call.
  // Define in IndexVarRel a function that takes in an Expr and produces an Expr for bound
  // for split: outer: Div(expr, splitfactor), Div(expr, splitfactor), inner: 0, splitfactor
  // what about for reordered split: same loop bounds just reordered loops (this might change for different tail strategies)

  if (isUnderived(indexVar)) {
    taco_iassert(underivedBounds.count(indexVar) == 1);
    return underivedBounds[indexVar];
  }

  std::vector<IndexVar> derivedVarOrderExceptLast = derivedVarOrder;
  if (!derivedVarOrderExceptLast.empty()) {
    derivedVarOrderExceptLast.pop_back();
  }
  taco_iassert(std::find(derivedVarOrderExceptLast.begin(), derivedVarOrderExceptLast.end(), indexVar) == derivedVarOrderExceptLast.end());

  std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds;
  std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds;
  for (const IndexVar& parent : getParents(indexVar)) {
    parentIterBounds[parent] = deriveIterBounds(parent, derivedVarOrder, underivedBounds, variableNames, iterators);
    vector<IndexVar> underivedParentAncestors = getUnderivedAncestors(parent);
    // TODO: this is okay for now because we don't need parentCoordBounds for fused taco_iassert(underivedParentAncestors.size() == 1);
    IndexVar underivedParent = underivedParentAncestors[0];
    parentCoordBounds[parent] = deriveCoordBounds(derivedVarOrderExceptLast, underivedBounds, variableNames, iterators)[underivedParent];
  }

  IndexVarRel rel = parentRelMap.at(indexVar);
  return rel.getNode()->deriveIterBounds(indexVar, parentIterBounds, parentCoordBounds, variableNames, iterators, *this);
}

bool ProvenanceGraph::hasCoordBounds(IndexVar indexVar) const {
  return !isUnderived(indexVar) && isCoordVariable(indexVar);
}

// position variable if any pos relationship parent
bool ProvenanceGraph::isPosVariable(taco::IndexVar indexVar) const {
  if (isUnderived(indexVar)) return false;
  if (parentRelMap.at(indexVar).getRelType() == POS) return true;
  for (const IndexVar& parent : getParents(indexVar)) {
    if (isPosVariable(parent)) {
      return true;
    }
  }
  return false;
}

bool ProvenanceGraph::isPosOfAccess(IndexVar indexVar, Access access) const {
  if (isUnderived(indexVar)) return false;
  if (parentRelMap.at(indexVar).getRelType() == POS) {
    return equals(parentRelMap.at(indexVar).getNode<PosRelNode>()->getAccess(), access);
  }
  else if (parentRelMap.at(indexVar).getRelType() == FUSE) {
    return false; // lose pos of access status through fuse
  }
  for (const IndexVar& parent : getParents(indexVar)) {
    if (isPosOfAccess(parent, access)) {
      return true;
    }
  }
  return false;
}

bool ProvenanceGraph::hasPosDescendant(taco::IndexVar indexVar) const {
  if (isPosVariable(indexVar)) return true;
  if (isFullyDerived(indexVar)) return false;
  IndexVarRel rel = childRelMap.at(indexVar);
  if (rel.getRelType() == FUSE) {
    vector<IndexVar> partners = getParents(getChildren(indexVar)[0]);
    if ((indexVar == partners[0] && isPosVariable(partners[1])) || (indexVar == partners[1] && isPosVariable(partners[0]))) {
      // can't get pos descendant from being fused with an already pos variable (need to be turned pos along derivation path)
      return false;
    }
  }
  for (auto child : getChildren(indexVar)) {
    if (hasPosDescendant(child)) return true;
  }
  return false;
}

bool ProvenanceGraph::isCoordVariable(taco::IndexVar indexVar) const {
  return !isPosVariable(indexVar);
}

bool ProvenanceGraph::hasExactBound(IndexVar indexVar) const {
  if (isUnderived(indexVar)) {
    return false;
  }

  IndexVarRel rel = parentRelMap.at(indexVar);
  if(rel.getRelType() == BOUND)
  {
    return rel.getNode<BoundRelNode>()->getBoundType() == BoundType::MaxExact;
  }
  // TODO: include non-irregular variables
  return false;
}

std::vector<IndexVar> ProvenanceGraph::newlyRecoverableParents(taco::IndexVar indexVar,
                                                   std::set<taco::IndexVar> previouslyDefined) const {
  // for each parent is it not recoverable with previouslyDefined, but yes with previouslyDefined+indexVar
  if (isUnderived(indexVar)) {
    return {};
  }

  std::set<taco::IndexVar> defined = previouslyDefined;
  defined.insert(indexVar);

  std::vector<IndexVar> newlyRecoverable;

  for (const IndexVar& parent : getParents(indexVar)) {
    if (parentRelMap.at(indexVar).getRelType() == FUSE) {
      IndexVar irregularDescendant;
      if (getIrregularDescendant(indexVar, &irregularDescendant) && isPosVariable(irregularDescendant) && isCoordVariable(parent)) { // Fused Pos case needs to be tracked with special while loop
        if (parent == getParents(indexVar)[0]) {
          continue;
        }
      }
    }

    if (!isRecoverable(parent, previouslyDefined) && isRecoverable(parent, defined)) {
      newlyRecoverable.push_back(parent);
      std::vector<IndexVar> parentRecoverable = newlyRecoverableParents(parent, previouslyDefined);
      newlyRecoverable.insert(newlyRecoverable.end(), parentRecoverable.begin(), parentRecoverable.end());
    }
  }
  return newlyRecoverable;
}

std::vector<IndexVar> ProvenanceGraph::derivationPath(taco::IndexVar ancestor, taco::IndexVar indexVar) const {
  if (ancestor == indexVar) {
    return {indexVar};
  }

  for (IndexVar child : getChildren(ancestor)) {
    std::vector<IndexVar> childResult = derivationPath(child, indexVar);
    if (!childResult.empty()) {
      childResult.insert(childResult.begin(), ancestor);
      return childResult;
    }
  }
  // wrong path taken
  return {};
}

ir::Expr ProvenanceGraph::recoverVariable(taco::IndexVar indexVar,
                                           std::vector<IndexVar> definedVarOrder,
                                           std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                                           std::map<taco::IndexVar, taco::ir::Expr> childVariables,
                                           Iterators iterators) const {
  if (isFullyDerived(indexVar)) {
    return ir::Expr();
  }

  IndexVarRel rel = childRelMap.at(indexVar);

  std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds;
  std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds;
  for (IndexVar parent : rel.getNode()->getParents()) {
    vector<IndexVar> underivedParentAncestors = getUnderivedAncestors(parent);
    //TODO: taco_iassert(underivedParentAncestors.size() == 1);
    IndexVar underivedParent = underivedParentAncestors[0];
    parentIterBounds[parent] = deriveIterBounds(parent, definedVarOrder, underivedBounds, childVariables, iterators);
    parentCoordBounds[parent] = deriveCoordBounds(definedVarOrder, underivedBounds, childVariables, iterators)[underivedParent];
  }

  return rel.getNode()->recoverVariable(indexVar, childVariables, iterators, parentIterBounds, parentCoordBounds, *this);
}

ir::Stmt ProvenanceGraph::recoverChild(taco::IndexVar indexVar,
                                        std::map<taco::IndexVar, taco::ir::Expr> relVariables, bool emitVarDecl, Iterators iterators) const {
  if (isUnderived(indexVar)) {
    return ir::Stmt();
  }

  IndexVarRel rel = parentRelMap.at(indexVar);
  return rel.getNode()->recoverChild(indexVar, relVariables, emitVarDecl, iterators, *this);
}

std::set<IndexVar> ProvenanceGraph::getAllIndexVars() const {
  return nodes;
}

bool ProvenanceGraph::isDivided(IndexVar indexVar) const {
  // See if the indexVar has any children. If so, look at the relation that
  // created the parent-child relationship. If it is a divide, return true.
  auto children = this->getChildren(indexVar);
  if (children.size() > 0) {
    auto rel = this->childRelMap.at(indexVar);
    if (rel.getRelType() == DIVIDE) {
      return true;
    }
  }
  return false;
}

}
