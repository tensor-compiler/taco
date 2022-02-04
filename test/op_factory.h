#ifndef TACO_OP_FACTORY_H
#define TACO_OP_FACTORY_H

#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/index_notation.h"
#include "taco/ir/ir.h"


namespace taco {

// Algebras
struct BC_BD_CD {
  IterationAlgebra operator()(const std::vector<IndexExpr> &v) {
    IterationAlgebra r1 = Intersect(v[0], v[1]);
    IterationAlgebra r2 = Intersect(v[0], v[2]);
    IterationAlgebra r3 = Intersect(v[1], v[2]);

    IterationAlgebra omit = Complement(Intersect(Intersect(v[0], v[1]), v[2]));
    return Intersect(Union(Union(r1, r2), r3), omit);
  }
};

struct ComplementUnion {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    taco_iassert(regions.size() >= 2);
    IterationAlgebra unions = Complement(regions[0]);
    for(size_t i = 1; i < regions.size(); ++i) {
      unions = Union(unions, regions[i]);
    }
    return unions;
  }
};

struct IntersectGen {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    if (regions.size() < 2) {
      return IterationAlgebra();
    }

    IterationAlgebra intersections = regions[0];
    for(size_t i = 1; i < regions.size(); ++i) {
      intersections = Intersect(intersections, regions[i]);
    }
    return intersections;
  }
};

struct ComplementIntersect {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    if (regions.size() < 2) {
      return IterationAlgebra();
    }

    IterationAlgebra intersections = Complement(regions[0]);
    for(size_t i = 1; i < regions.size(); ++i) {
      intersections = Intersect(intersections, regions[i]);
    }
    return intersections;
  }
};

struct IntersectGenDeMorgan {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    IterationAlgebra unions = Complement(regions[0]);
    for(size_t i = 1; i < regions.size(); ++i) {
      unions = Union(unions, Complement(regions[i]));
    }
    return Complement(unions);
  }
};

struct xorGen {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    IterationAlgebra noIntersect = Complement(Intersect(regions[0], regions[1]));
    return Intersect(noIntersect, Union(regions[0], regions[1]));
  }
};

struct fullSpaceGen {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Union(Complement(regions[0]), regions[0]);
  }
};

struct emptyGen {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Intersect(Complement(regions[0]), regions[0]);
  }
};

struct intersectEdge {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    std::vector<IndexExpr> r = regions;
    return Intersect(Complement(Intersect(r[0], r[1])), Intersect(r[0], r[1]));
  }
};

struct unionEdge {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    std::vector<IndexExpr> r = regions;
    return Union(Complement(Intersect(r[0], r[1])), Intersect(r[0], r[1]));
  }
};

struct BfsMaskAlg {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    std::vector<IndexExpr> r = regions;
    return Intersect(r[0], Complement(r[1]));
  }
};

// Lowerers
struct MulAdd {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return ir::Add::make(ir::Mul::make(v[0], v[1]), v[2]);
  }
};

struct identityFunc {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return v[0];
  }
};

struct GeneralAdd {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() >= 2) << "Add operator needs at least two operands";
    ir::Expr add = ir::Add::make(v[0], v[1]);

    for (size_t idx = 2; idx < v.size(); ++idx) {
      add = ir::Add::make(add, v[idx]);
    }

    return add;
  }
};

struct MinImpl {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() >= 2) << "Min operator needs at least two operands";
    return ir::Min::make(v[0], v[1]);
  }
};

struct BfsLower {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return v[0];
  }
};

struct OrImpl {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return ir::Or::make(v[0], v[1]);
  }
};

struct BitOrImpl {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return ir::BitOr::make(v[0], v[1]);
  }
};

struct AndImpl {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return ir::And::make(v[0], v[1]);
  }
};

// Special definitions
struct MulRegionDef {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Add operator needs at least two operands";
    return ir::Mul::make(v[0], v[1]);
  }
};

struct SubRegionDef {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 2) << "Sub def needs two operands";
    return ir::Sub::make(v[1], v[0]);
  }
};


}
#endif //TACO_OP_FACTORY_H
