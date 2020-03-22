#ifndef TACO_OP_FACTORY_H
#define TACO_OP_FACTORY_H

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

struct UnionDeMorgan {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    if(regions.empty()) {
      return IterationAlgebra();
    }

    if (regions.size() == 1) {
      return regions[0];
    }

    IterationAlgebra intersections = Complement(regions[0]);
    for(size_t i = 1; i < regions.size(); ++i) {
      intersections = Intersect(intersections, Complement(regions[i]));
    }
    return Complement(intersections);
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
    IterationAlgebra unions;
    for(const auto& region : regions) {
      unions = Union(unions, Complement(region));
    }
    return Complement(unions);
  }
};


// Lowerers
struct MulAdd {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    return ir::Add::make(ir::Mul::make(v[0], v[1]), v[2]);
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
