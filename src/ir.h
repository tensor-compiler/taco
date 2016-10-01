#ifndef TAC_IR_H
#define TAC_IR_H

#include "util/intrusive_ptr.h"
#include "util/uncopyable.h"
#include "component_types.h"

namespace taco {
namespace internal {

/** All IR nodes get unique IDs for RTTI */
enum class IRNodeType {
  Literal,
  Var,
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Min,
  Max,
  Not,
  Eq,
  Ne,
  Gt,
  Lt,
  Ge,
  And,
  Or,
  IfThenElse,
  Load,
  Store,
  For,
  Block
};

/** Base class for backend IR */
struct IRNode : private util::Uncopyable {
  IRNode() {}
  virtual ~IRNode() {}
  /** Each IRNode subclasses carries a unique pointer we use to determine
   * its node type, because compiler RTTI sucks.
   */
  virtual IRNodeType type_info() const = 0;

  
  mutable long ref = 0;
  friend void aquire(const IRNode* node) { (node->ref)++; }
  friend void release(const IRNode* node) { if (--(node->ref)) delete node; }
};

/** Base class for statements. */
struct BaseStmtNode : public IRNode {
};

/** Base class for expression nodes, which have a type. */
struct BaseExprNode : public IRNode {
  ComponentType type = typeOf<double>();
};

/** Use the "curiously recurring template pattern" from Halide
 * to avoid duplicated code in IR nodes.  This provides the type
 * info for each class (and will handle visitor accept methods as
 * well).
 */
template<typename T>
struct ExprNode : public BaseExprNode {
  virtual IRNodeType type_info() const { return T::_type_info; }
};

template <typename T>
struct StmtNode : public BaseStmtNode {
  virtual IRNodeType type_info() const { return T::_type_info; }
};


/** IR nodes are passed around using opaque handles.  This class 
 * handles type conversion, and will handle visitors.
 */
struct IRHandle : public util::IntrusivePtr<const IRNode> {
  IRHandle() : util::IntrusivePtr<const IRNode>() {}
  IRHandle(const IRNode *p) : util::IntrusivePtr<const IRNode>(p) {}
  
  /** Cast this IR node to its actual type. */
  template <typename T> const T *as() const {
    if (ptr && ptr->type_info() == T::_type_info) {
      return (const T*)ptr;
    } else {
      return nullptr;
    }
  }
};

/** An expression. */
class Expr : public IRHandle {
public:
  Expr() : IRHandle() {}
  Expr(const BaseExprNode *expr) : IRHandle(expr) {}
  
  /** Get the type of this expression node */
  ComponentType type() const {
    return ((const BaseExprNode *)ptr)->type;
  }
};

/** A statement. */
class Stmt : public IRHandle {
public:
  Stmt() : IRHandle() {}
  Stmt(const BaseStmtNode* stmt) : IRHandle(stmt) {}
};

std::ostream &operator<<(std::ostream &os, const Stmt &);
std::ostream &operator<<(std::ostream &os, const Expr &);

// Actual nodes start here

/** A literal. */
struct Literal : public ExprNode<Literal> {
public:
  int64_t value;
  
  static Expr make(int val);
  static Expr make(double val, ComponentType type=ComponentType::Double);
  
  static const IRNodeType _type_info = IRNodeType::Literal;
};

/** A variable.  We probably have to distinguish between pointer types
 * etc., but for now this is always assumed to be a pointer type.
 */
struct Var : public ExprNode<Var> {
public:
  std::string name;
  
  static Expr make(std::string name, ComponentType type);

  static const IRNodeType _type_info = IRNodeType::Var;
};

/** Addition. */
struct Add : public ExprNode<Add> {
public:
  Expr a;
  Expr b;
  
  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, ComponentType type);
  
  static const IRNodeType _type_info = IRNodeType::Add;
};

/** Subtraction. */
struct Sub : public ExprNode<Sub> {
public:
  Expr a;
  Expr b;
  
  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, ComponentType type);
  
  static const IRNodeType _type_info = IRNodeType::Sub;
};


} // namespace internal
} // namespace tac

#endif
