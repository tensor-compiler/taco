#ifndef TACO_IR_H
#define TACO_IR_H

#include <vector>
#include "taco/format.h"

#include "taco/type.h"
#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/uncopyable.h"

namespace taco {
namespace ir {

class IRVisitorStrict;

/** All IR nodes get unique IDs for RTTI */
enum class IRNodeType {
  Literal,
  Var,
  Neg,
  Sqrt,
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Min,
  Max,
  BitAnd,
  Not,
  Eq,
  Neq,
  Gt,
  Lt,
  Gte,
  Lte,
  And,
  Or,
  IfThenElse,
  Case,
  Load,
  Store,
  For,
  While,
  Block,
  Scope,
  Function,
  VarAssign,
  Allocate,
  Comment,
  BlankLine,
  Print,
  GetProperty
};

enum class TensorProperty {
  Order,
  Dimension,
  ComponentSize,
  ModeOrdering,
  ModeTypes,
  Indices,
  Values
};

/** Base class for backend IR */
struct IRNode : private util::Uncopyable {
  IRNode() {}
  virtual ~IRNode() {}
  virtual void accept(IRVisitorStrict *v) const = 0;
  
  /** Each IRNode subclasses carries a unique pointer we use to determine
   * its node type, because compiler RTTI sucks.
   */
  virtual IRNodeType type_info() const = 0;

  mutable long ref = 0;
  friend void acquire(const IRNode* node) {
    ++(node->ref);
  }
  friend void release(const IRNode* node) {
    if (--(node->ref) == 0) {
      delete node;
    }
  }
};

/** Base class for statements. */
struct BaseStmtNode : public IRNode {
};

/** Base class for expression nodes, which have a type. */
struct BaseExprNode : public IRNode {
  Type type = Type(Type::Float, 64);
};

/** Use the "curiously recurring template pattern" from Halide
 * to avoid duplicated code in IR nodes.  This provides the type
 * info for each class (and will handle visitor accept methods as
 * well).
 */
template<typename T>
struct ExprNode : public BaseExprNode {
  virtual ~ExprNode() = default;
  void accept(IRVisitorStrict *v) const;
  virtual IRNodeType type_info() const { return T::_type_info; }
};

template <typename T>
struct StmtNode : public BaseStmtNode {
  virtual ~StmtNode() = default;
  void accept(IRVisitorStrict *v) const;
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
  
  /** Dispatch to the corresponding visitor method */
  void accept(IRVisitorStrict *v) const {
    ptr->accept(v);
  }
};

/** An expression. */
class Expr : public IRHandle {
public:
  Expr() : IRHandle() {}
  Expr(int);
  Expr(float);
  Expr(double);

  Expr(const BaseExprNode *expr) : IRHandle(expr) {}

  /** Get the type of this expression node */
  Type type() const {
    return ((const BaseExprNode *)ptr)->type;
  }
};

/** This is a custom comparator that allows
 * Exprs to be used in a map.  Inspired by Halide.
 */
class ExprCompare {
public:
  bool operator()(Expr a, Expr b) const { return a.ptr < b.ptr; }
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
  double dbl_value;

  static Expr make(bool val);
  static Expr make(int val);
  static Expr make(double val, Type type=Type(Type::Float, 64));

  static const IRNodeType _type_info = IRNodeType::Literal;
};

/** A variable.  */
struct Var : public ExprNode<Var> {
public:
  std::string name;
  bool is_ptr;
  bool is_tensor;
  Format format;

  static Expr make(std::string name, Type type, bool is_ptr=false);
  static Expr make(std::string name, Type type, Format format);

  static const IRNodeType _type_info = IRNodeType::Var;
};


/** Negation */
struct Neg : public ExprNode<Neg> {
public:
  Expr a;
  
  static Expr make(Expr a);
  
  static const IRNodeType _type_info = IRNodeType::Neg;
};

/** A square root */
struct Sqrt : public ExprNode<Sqrt> {
public:
  Expr a;
  
  static Expr make(Expr a);
  
  static const IRNodeType _type_info = IRNodeType::Sqrt;
};

/** Addition. */
struct Add : public ExprNode<Add> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);

  static const IRNodeType _type_info = IRNodeType::Add;
};

/** Subtraction. */
struct Sub : public ExprNode<Sub> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);

  static const IRNodeType _type_info = IRNodeType::Sub;
};

/** Multiplication. */
struct Mul : public ExprNode<Mul> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);

  static const IRNodeType _type_info = IRNodeType::Mul;
};

/** Division. */
struct Div : public ExprNode<Div> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);

  static const IRNodeType _type_info = IRNodeType::Div;
};

/** Remainder. */
struct Rem : public ExprNode<Rem> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);

  static const IRNodeType _type_info = IRNodeType::Rem;
};

/** Minimum of two values. */
struct Min : public ExprNode<Min> {
public:
  std::vector<Expr> operands;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);
  static Expr make(std::vector<Expr> operands);
  static Expr make(std::vector<Expr> operands, Type type);

  static const IRNodeType _type_info = IRNodeType::Min;
};

/** Maximum of two values. */
struct Max : public ExprNode<Max> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Type type);

  static const IRNodeType _type_info = IRNodeType::Max;
};

/** Bitwise and: a & b */
struct BitAnd : public ExprNode<BitAnd> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::BitAnd;
};

/** Equality: a==b. */
struct Eq : public ExprNode<Eq> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Eq;
};

/** Inequality: a!=b. */
struct Neq : public ExprNode<Neq> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Neq;
};

/** Greater than: a > b. */
struct Gt : public ExprNode<Gt> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Gt;
};

/** Less than: a < b. */
struct Lt : public ExprNode<Lt> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Lt;
};

/** Greater than or equal: a >= b. */
struct Gte : public ExprNode<Gte> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Gte;
};

/** Less than or equal: a <= b. */
struct Lte : public ExprNode<Lte> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Lte;
};

/** And: a && b. */
struct And : public ExprNode<And> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::And;
};

/** Or: a || b. */
struct Or : public ExprNode<Or> {
public:
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Or;
};

/** A load from an array: arr[loc]. */
struct Load : public ExprNode<Load> {
public:
  Expr arr;
  Expr loc;

  static Expr make(Expr arr);
  static Expr make(Expr arr, Expr loc);

  static const IRNodeType _type_info = IRNodeType::Load;
};

/** A sequence of statements. */
struct Block : public StmtNode<Block> {
public:
  std::vector<Stmt> contents;
  void append(Stmt stmt) { contents.push_back(stmt); }

  static Stmt make();
  static Stmt make(std::vector<Stmt> b);

  static const IRNodeType _type_info = IRNodeType::Block;
};

/** A variable scope. */
struct Scope : public StmtNode<Scope> {
public:
  Stmt scopedStmt;

  static Stmt make(Stmt scopedStmt);

  static const IRNodeType _type_info = IRNodeType::Scope;
};

/** A store to an array location: arr[loc] = data */
struct Store : public StmtNode<Store> {
public:
  Expr arr;
  Expr loc;
  Expr data;

  static Stmt make(Expr arr, Expr loc, Expr data);

  static const IRNodeType _type_info = IRNodeType::Store;
};

/** A conditional statement. */
struct IfThenElse : public StmtNode<IfThenElse> {
public:
  Expr cond;
  Stmt then;
  Stmt otherwise;
  
  static Stmt make(Expr cond, Stmt then);
  static Stmt make(Expr cond, Stmt then, Stmt otherwise);
  
  static const IRNodeType _type_info = IRNodeType::IfThenElse;
};

/** A series of conditionals. */
struct Case : public StmtNode<Case> {
public:
  std::vector<std::pair<Expr,Stmt>> clauses;
  bool alwaysMatch;
  
  static Stmt make(std::vector<std::pair<Expr,Stmt>> clauses, bool alwaysMatch);
  
  static const IRNodeType _type_info = IRNodeType::Case;
};

enum class LoopKind {Serial, Static, Dynamic, Vectorized};

/** A for loop from start to end by increment.
 * A vectorized loop will require the increment to be 1 and the
 * end to be (start + Literal) or possibly (start + Var).
 *
 * If the loop is vectorized, the width says which vector width
 * to use.  By default (0), it will not set a specific width and
 * let clang determine the width to use.
 */
struct For : public StmtNode<For> {
public:
  Expr var;
  Expr start;
  Expr end;
  Expr increment;
  Stmt contents;
  LoopKind kind;
  int vec_width;  // vectorization width
  
  static Stmt make(Expr var, Expr start, Expr end, Expr increment,
                   Stmt contents, LoopKind kind=LoopKind::Serial,
                   int vec_width=0);
  
  static const IRNodeType _type_info = IRNodeType::For;
};

/** A while loop.  We prefer For loops when possible, but
 * these are necessary for merging.
 */
struct While : public StmtNode<While> {
  Expr cond;
  Stmt contents;
  LoopKind kind;
  int vec_width;  // vectorization width
  
  static Stmt make(Expr cond, Stmt contents, LoopKind kind=LoopKind::Serial,
    int vec_width=0);
  
  static const IRNodeType _type_info = IRNodeType::While;
};

/** Top-level function for codegen */
struct Function : public StmtNode<Function> {
public:
  std::string name;
  Stmt body;
  std::vector<Expr> inputs;
  std::vector<Expr> outputs;
  
  static Stmt make(std::string name, std::vector<Expr> inputs,
                   std::vector<Expr> outputs, Stmt body);
  
  static const IRNodeType _type_info = IRNodeType::Function;
};
  
/** Assigning a Var to an expression */
struct VarAssign : public StmtNode<VarAssign> {
public:
  Expr lhs;   // must be a Var
  Expr rhs;
  bool is_decl;
  
  static Stmt make(Expr lhs, Expr rhs, bool is_decl=false);
  
  static const IRNodeType _type_info = IRNodeType::VarAssign;
};

/** An Allocate node that allocates some memory for a Var */
struct Allocate : public StmtNode<Allocate> {
public:
  Expr var;   // must be a Var
  Expr num_elements;
  bool is_realloc;
  
  static Stmt make(Expr var, Expr num_elements, bool is_realloc=false);
  
  static const IRNodeType _type_info = IRNodeType::Allocate;
};

/** A comment */
struct Comment : public StmtNode<Comment> {
public:
  std::string text;
  
  static Stmt make(std::string text);
  
  static const IRNodeType _type_info = IRNodeType::Comment;
};

/** A blank statement (no-op) */
struct BlankLine : public StmtNode<BlankLine> {
public:
  static Stmt make();

  static const IRNodeType _type_info = IRNodeType::BlankLine;
};

/** A print statement.
 * Takes in a printf-style format string and Exprs to pass
 * for the values.
 */
struct Print : public StmtNode<Print> {
public:
  std::string fmt;
  std::vector<Expr> params;
  
  static Stmt make(std::string fmt, std::vector<Expr> params={});
  
  static const IRNodeType _type_info = IRNodeType::Print;
};

/** A tensor property.
 * This unpacks one of the properties of a tensor into an Expr.
 */
struct GetProperty : public ExprNode<GetProperty> {
public:
  Expr tensor;
  TensorProperty property;
  int mode;
  int index = 0;
  std::string name;

  static Expr make(Expr tensor, TensorProperty property, int mode=0);
  static Expr make(Expr tensor, TensorProperty property, int mode,
                   int index, std::string name);
  
  static const IRNodeType _type_info = IRNodeType::GetProperty;
};

template <typename E>
inline bool isa(Expr e) {
  return e.defined() && dynamic_cast<const E*>(e.ptr) != nullptr;
}

template <typename S>
inline bool isa(Stmt s) {
  return s.defined() && dynamic_cast<const S*>(s.ptr) != nullptr;
}

template <typename E>
inline const E* to(Expr e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " <<typeid(E).name();
  return static_cast<const E*>(e.ptr);
}

template <typename S>
inline const S* to(Stmt s) {
  taco_iassert(isa<S>(s)) <<
      "Cannot convert " << typeid(s).name() << " to " <<typeid(S).name();
  return static_cast<const S*>(s.ptr);
}

}}
#endif
