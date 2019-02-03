#include "test.h"

#include "taco/ir/ir.h"
#include "taco/ir/simplify.h"

using taco::ir::Var;
using taco::ir::VarDecl;
using taco::ir::Assign;
using taco::ir::Block;
using taco::Int32;
using taco::ir::simplify;
using taco::ir::Neg;
using taco::ir::Add;

TEST(expr, simplify_copy) {
  auto a = Var::make("a", Int32), 
       b = Var::make("b", Int32),
       c = Var::make("c", Int32),
       d = Var::make("d", Int32);

  auto aDecl = VarDecl::make(a, 42),
       bDecl = VarDecl::make(b, a),
       cDecl = VarDecl::make(c, b),
       // assign `d' a non-trivial expression so it won't get optimized away
       dDecl = VarDecl::make(d, Neg::make(c)),
       dInc = Assign::make(d, Add::make(d, 1));
  auto block = Block::make({aDecl, bDecl, cDecl, dDecl, dInc});

  auto simplified = simplify(block);
  auto *simplifiedBlock = simplified.as<Block>();
  ASSERT_EQ(simplifiedBlock->contents.size(), size_t(3));

  const VarDecl *simplifiedDDecl = simplifiedBlock->contents[1].as<VarDecl>();
  ASSERT_EQ(simplifiedDDecl->rhs.as<Neg>()->a, a);
}
