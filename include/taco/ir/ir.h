#ifndef TACO_IR_H
#define TACO_IR_H

#include <vector>
#include <typeinfo>
#include <utility>

#include "taco/type.h"
#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/uncopyable.h"
#include "taco/storage/typed_value.h"
#include <cstring>

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
  BitOr,
  Not,
  Eq,
  Neq,
  Gt,
  Lt,
  Gte,
  Lte,
  And,
  Or,
  Cast,
  Call,
  IfThenElse,
  Case,
  Switch,
  Load,
  Malloc,
  Sizeof,
  Store,
  For,
  While,
  Block,
  Scope,
  Function,
  VarDecl,
  VarAssign,
  Yield,
  Allocate,
  Free,
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
  Values,
  ValuesSize
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
  Datatype type = Float();
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

  Expr(bool);
  Expr(int8_t);
  Expr(int16_t);
  Expr(int32_t);
  Expr(int64_t);
  Expr(uint8_t);
  Expr(uint16_t);
  Expr(uint32_t);
  Expr(uint64_t);
  Expr(float);
  Expr(double);
  Expr(std::complex<float>);
  Expr(std::complex<double>);

  Expr(const BaseExprNode *expr) : IRHandle(expr) {}

  /** Get the type of this expression node */
  Datatype type() const {
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
  TypedComponentPtr value;

  static Expr make(TypedComponentVal val, Datatype type) {
    taco_iassert(isScalar(type));
    Literal *lit = new Literal;
    lit->type = type;
    lit->value = TypedComponentPtr(type, malloc(type.getNumBytes()));
    *(lit->value) = val;
    return lit;
  }

  template <typename T>
  static Expr make(T val, Datatype type) {
    return make(TypedComponentVal(type, &val), type);
  }

  template <typename T>
  static Expr make(T val) {
    return make(val, taco::type<T>());
  }

  /// Returns a zero literal of the given type.
  static Expr zero(Datatype datatype);

  ~Literal();

  template <typename T>
  T getValue() const {
    taco_iassert(taco::type<T>() == type);
    return *static_cast<const T*>(value.get());
  }

  TypedComponentVal getTypedVal() const {
    return *value;
  }

  bool getBoolValue() const;
  int64_t getIntValue() const;
  uint64_t getUIntValue() const;
  double getFloatValue() const;
  std::complex<double> getComplexValue() const;

  static const IRNodeType _type_info = IRNodeType::Literal;

  bool equalsScalar(double scalar) const;
};


/** A variable.  */
struct Var : public ExprNode<Var> {
  std::string name;
  bool is_ptr;
  bool is_tensor;

  static Expr make(std::string name, Datatype type, bool is_ptr=false, 
                   bool is_tensor=false);

  static const IRNodeType _type_info = IRNodeType::Var;
};


/** Negation */
struct Neg : public ExprNode<Neg> {
  Expr a;
  
  static Expr make(Expr a);
  
  static const IRNodeType _type_info = IRNodeType::Neg;
};

/** A square root */
struct Sqrt : public ExprNode<Sqrt> {
  Expr a;
  
  static Expr make(Expr a);
  
  static const IRNodeType _type_info = IRNodeType::Sqrt;
};

/** Addition. */
struct Add : public ExprNode<Add> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Add;
};

/** Subtraction. */
struct Sub : public ExprNode<Sub> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Sub;
};

/** Multiplication. */
struct Mul : public ExprNode<Mul> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Mul;
};

/** Division. */
struct Div : public ExprNode<Div> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Div;
};

/** Remainder. */
struct Rem : public ExprNode<Rem> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Rem;
};

/** Minimum of two values. */
struct Min : public ExprNode<Min> {
  std::vector<Expr> operands;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);
  static Expr make(std::vector<Expr> operands);
  static Expr make(std::vector<Expr> operands, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Min;
};

/** Maximum of two values. */
struct Max : public ExprNode<Max> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);
  static Expr make(Expr a, Expr b, Datatype type);

  static const IRNodeType _type_info = IRNodeType::Max;
};

/** Bitwise and: a & b */
struct BitAnd : public ExprNode<BitAnd> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::BitAnd;
};

/** Bitwise or: a | b */
struct BitOr : public ExprNode<BitOr> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::BitOr;
};

/** Equality: a==b. */
struct Eq : public ExprNode<Eq> {
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
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Gt;
};

/** Less than: a < b. */
struct Lt : public ExprNode<Lt> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Lt;
};

/** Greater than or equal: a >= b. */
struct Gte : public ExprNode<Gte> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Gte;
};

/** Less than or equal: a <= b. */
struct Lte : public ExprNode<Lte> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Lte;
};

/** And: a && b. */
struct And : public ExprNode<And> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::And;
};

/** Or: a || b. */
struct Or : public ExprNode<Or> {
  Expr a;
  Expr b;

  static Expr make(Expr a, Expr b);

  static const IRNodeType _type_info = IRNodeType::Or;
};

/** Type cast. */
struct Cast : public ExprNode<Cast> {
  Expr a;

  static Expr make(Expr a, Datatype newType);

  static const IRNodeType _type_info = IRNodeType::Cast;
};

/** A call of a function. */
struct Call : public ExprNode<Call> {
  std::string func;
  std::vector<Expr> args;

  static Expr make(const std::string& func, const std::vector<Expr>& args, 
                   Datatype type);

  static const IRNodeType _type_info = IRNodeType::Call;
};

/** A load from an array: arr[loc]. */
struct Load : public ExprNode<Load> {
  Expr arr;
  Expr loc;

  static Expr make(Expr arr);
  static Expr make(Expr arr, Expr loc);

  static const IRNodeType _type_info = IRNodeType::Load;
};

/** Allocate size bytes of memory */
struct Malloc : public ExprNode<Malloc> {
public:
  Expr size;

  static Expr make(Expr size);

  static const IRNodeType _type_info = IRNodeType::Malloc;
};

/** Compute the size of a type */
struct Sizeof : public ExprNode<Sizeof> {
public:
  Type sizeofType;

  static Expr make(Type type);

  static const IRNodeType _type_info = IRNodeType::Sizeof;
};

/** A sequence of statements. */
struct Block : public StmtNode<Block> {
  std::vector<Stmt> contents;
  void append(Stmt stmt) { contents.push_back(stmt); }

  static Stmt make();

  static Stmt make(std::vector<Stmt> stmts);
  template <typename... Stmts>
  static Stmt make(const Stmts&... stmts) {
    return make({stmts...});
  }

  /// Create a block with blank lines between statements.
  static Stmt blanks(std::vector<Stmt> stmts);
  template <typename... Stmts>
  static Stmt blanks(const Stmts&... stmts) {
    return blanks({stmts...});
  }

  static const IRNodeType _type_info = IRNodeType::Block;
};

/** A variable scope. */
struct Scope : public StmtNode<Scope> {
  Stmt scopedStmt;

  static Stmt make(Stmt scopedStmt);

  static const IRNodeType _type_info = IRNodeType::Scope;
};

/** A store to an array location: arr[loc] = data */
struct Store : public StmtNode<Store> {
  Expr arr;
  Expr loc;
  Expr data;

  static Stmt make(Expr arr, Expr loc, Expr data);

  static const IRNodeType _type_info = IRNodeType::Store;
};

/** A conditional statement. */
struct IfThenElse : public StmtNode<IfThenElse> {
  Expr cond;
  Stmt then;
  Stmt otherwise;
  
  static Stmt make(Expr cond, Stmt then);
  static Stmt make(Expr cond, Stmt then, Stmt otherwise);
  
  static const IRNodeType _type_info = IRNodeType::IfThenElse;
};

/** A series of conditionals. */
struct Case : public StmtNode<Case> {
  std::vector<std::pair<Expr,Stmt>> clauses;
  bool alwaysMatch;
  
  static Stmt make(std::vector<std::pair<Expr,Stmt>> clauses, bool alwaysMatch);
  
  static const IRNodeType _type_info = IRNodeType::Case;
};

/** A switch statement. */
struct Switch : public StmtNode<Switch> {
  std::vector<std::pair<Expr,Stmt>> cases;
  Expr controlExpr;
  
  static Stmt make(std::vector<std::pair<Expr,Stmt>> cases, Expr controlExpr);
  
  static const IRNodeType _type_info = IRNodeType::Switch;
};

enum class LoopKind {Serial, Static, Dynamic, Runtime, Vectorized};

/** A for loop from start to end by increment.
 * A vectorized loop will require the increment to be 1 and the
 * end to be (start + Literal) or possibly (start + Var).
 *
 * If the loop is vectorized, the width says which vector width
 * to use.  By default (0), it will not set a specific width and
 * let clang determine the width to use.
 */
struct For : public StmtNode<For> {
  Expr var;
  Expr start;
  Expr end;
  Expr increment;
  Stmt contents;
  LoopKind kind;
  int vec_width;  // vectorization width
  bool accelerator;
  
  static Stmt make(Expr var, Expr start, Expr end, Expr increment,
                   Stmt contents, LoopKind kind=LoopKind::Serial,
                   bool accelerator=false, int vec_width=0);
  
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
  std::string name;
  Stmt body;
  std::vector<Expr> inputs;
  std::vector<Expr> outputs;
  
  static Stmt make(std::string name,
                   std::vector<Expr> outputs, std::vector<Expr> inputs,
                   Stmt body);
  
  std::pair<std::vector<Datatype>,Datatype> getReturnType() const;
  
  static const IRNodeType _type_info = IRNodeType::Function;
};

/** Declaring and initializing a Var */
struct VarDecl : public StmtNode<VarDecl> {
  Expr var;
  Expr rhs;

  static Stmt make(Expr var, Expr rhs);

  static const IRNodeType _type_info = IRNodeType::VarDecl;
};

/** Assigning a Var to an expression */
struct Assign : public StmtNode<Assign> {
  Expr lhs;
  Expr rhs;
  
  static Stmt make(Expr lhs, Expr rhs);
  
  static const IRNodeType _type_info = IRNodeType::VarAssign;
};

/** Yield a result component */
struct Yield : public StmtNode<Yield> {
  std::vector<Expr> coords;
  Expr val;

  static Stmt make(std::vector<Expr> coords, Expr val);

  static const IRNodeType _type_info = IRNodeType::Yield;
};

/** Allocate memory for a ptr var */
struct Allocate : public StmtNode<Allocate> {
  Expr var;
  Expr num_elements;
  Expr old_elements; // used for realloc in CUDA
  bool is_realloc;
  
  static Stmt make(Expr var, Expr num_elements, bool is_realloc=false,
                   Expr old_elements=Expr());
  
  static const IRNodeType _type_info = IRNodeType::Allocate;
};

/** Free memory for a ptr var */
struct Free : public StmtNode<Free> {
  Expr var;

  static Stmt make(Expr var);

  static const IRNodeType _type_info = IRNodeType::Free;
};

/** A comment */
struct Comment : public StmtNode<Comment> {
  std::string text;
  
  static Stmt make(std::string text);
  
  static const IRNodeType _type_info = IRNodeType::Comment;
};

/** A blank statement (no-op) */
struct BlankLine : public StmtNode<BlankLine> {
  static Stmt make();

  static const IRNodeType _type_info = IRNodeType::BlankLine;
};

/** A print statement.
 * Takes in a printf-style format string and Exprs to pass
 * for the values.
 */
struct Print : public StmtNode<Print> {
  std::string fmt;
  std::vector<Expr> params;
  
  static Stmt make(std::string fmt, std::vector<Expr> params={});
  
  static const IRNodeType _type_info = IRNodeType::Print;
};

/** A tensor property.
 * This unpacks one of the properties of a tensor into an Expr.
 */
struct GetProperty : public ExprNode<GetProperty> {
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

template<typename T>
bool isValue(Expr expr, T val) {
  if (isa<Literal>(expr)) {
    auto literal = to<Literal>(expr);
    if (literal->type == type<T>()) {
      return literal->getValue<T>() == val;
    }
  }
  return false;
}

}}
#endif
