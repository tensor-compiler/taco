#ifndef TACO_INDEX_NOTATION_H
#define TACO_INDEX_NOTATION_H

#include <functional>
#include <ostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <functional>

#include "taco/util/name_generator.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/type.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/index_notation_nodes_abstract.h"
#include "taco/ir_tags.h"
#include "taco/index_notation/provenance_graph.h"
#include "taco/index_notation/properties.h"

namespace taco {

class Type;
class Dimension;
class Format;
class Schedule;

class IndexVar;
class WindowedIndexVar;
class IndexSetVar;
class TensorVar;

class IndexStmt;
class IndexExpr;
class Assignment;
class Access;

class IterationAlgebra;

struct AccessNode;
struct IndexVarIterationModifier;
struct LiteralNode;
struct NegNode;
struct SqrtNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;
struct CastNode;
struct CallNode;
struct CallIntrinsicNode;
struct ReductionNode;
struct IndexVarNode;

struct AssignmentNode;
struct YieldNode;
struct ForallNode;
struct WhereNode;
struct SequenceNode;
struct AssembleNode;
struct MultiNode;
struct SuchThatNode;

class IndexExprVisitorStrict;
class IndexStmtVisitorStrict;

/// Describe the relation between indexVar sets of lhs and rhs in an Assignment node.
/// equal: lhs = rhs
/// none: lhs and rhs are mutually exclusive. And lhs and rhs are not empty sets.
/// lcr: rhs is a proper subset of lhs. (lhs contains rhs)
/// rcl: lhs is a proper subset of rhs. (rhs contains lhs)
/// inter: lhs and rhs share common elements but are not equal or empty. Some examples:
/// ```
/// // equal
/// ws(i1) += A(i1) // i1 is a child index node
/// ws(i) = A(i) // i is a parent index node
///
/// // none
/// ws(i1) += A(i) // i1 is a child of i
/// B_new(i) = B(i1)
///
/// // lcr
/// ws(i,k) = A(i) * B(i)
///
/// // rcl
/// ws(i) += A(i,k) * B(i,k)
///
/// // inter
/// ws(i,j) += A(i,k) * B(k,j)
/// ```
///
enum IndexSetRel {
    equal, none, lcr, rcl, inter
};

/// Return true if the index statement is of the given subtype.  The subtypes
/// are Assignment, Forall, Where, Sequence, and Multi.
template <typename SubType> bool isa(IndexExpr);

/// Casts the index statement to the given subtype. Assumes S is a subtype and
/// the subtypes are Assignment, Forall, Where, Sequence, and Multi.
template <typename SubType> SubType to(IndexExpr);

/// A tensor index expression describes a tensor computation as a scalar
/// expression where tensors are indexed by index variables (`IndexVar`).  The
/// index variables range over the tensor dimensions they index, and the scalar
/// expression is evaluated at every point in the resulting iteration space.
/// Index variables that are not used to index the result/left-hand-side are
/// called summation variables and are summed over. Some examples:
/// ```
/// // Matrix addition
/// A(i,j) = B(i,j) + C(i,j);
///
/// // Tensor addition (order-3 tensors)
/// A(i,j,k) = B(i,j,k) + C(i,j,k);
///
/// // Matrix-vector multiplication
/// a(i) = B(i,j) * c(j);
///
/// // Tensor-vector multiplication (order-3 tensor)
/// A(i,j) = B(i,j,k) * c(k);
///
/// // Matricized tensor times Khatri-Rao product (MTTKRP) from data analytics
/// A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
/// ```
///
/// @see IndexVar Index into index expressions.
/// @see TensorVar Operands of index expressions.
class IndexExpr : public util::IntrusivePtr<const IndexExprNode> {
public:
  IndexExpr() : util::IntrusivePtr<const IndexExprNode>(nullptr) {}
  IndexExpr(const IndexExprNode* n) : util::IntrusivePtr<const IndexExprNode>(n) {}

  /// Construct a scalar tensor access.
  /// ```
  /// A(i,j) = b;
  /// ```
  IndexExpr(TensorVar);

  /// Consturct an integer literal.
  /// ```
  /// A(i,j) = 1;
  /// ```
  IndexExpr(char);
  IndexExpr(int8_t);
  IndexExpr(int16_t);
  IndexExpr(int32_t);
  IndexExpr(int64_t);

  /// Consturct an unsigned integer literal.
  /// ```
  /// A(i,j) = 1u;
  /// ```
  IndexExpr(uint8_t);
  IndexExpr(uint16_t);
  IndexExpr(uint32_t);
  IndexExpr(uint64_t);

  /// Consturct double literal.
  /// ```
  /// A(i,j) = 1.0;
  /// ```
  IndexExpr(float);
  IndexExpr(double);

  /// Construct complex literal.
  /// ```
  /// A(i,j) = complex(1.0, 1.0);
  /// ```
  IndexExpr(std::complex<float>);
  IndexExpr(std::complex<double>);

  Datatype getDataType() const;

  /// Store the index expression's result to a dense workspace w.r.t. index
  /// variable `i` and replace the index expression (in the enclosing
  /// expression) with a workspace access expression.  The index variable `i` is
  /// retained in the enclosing expression and used to access the workspace,
  /// while `iw` replaces `i` in the index expression that computes workspace
  /// results.
  void workspace(IndexVar i, IndexVar iw, std::string name="");

  /// Store the index expression's result to a workspace of the given format
  /// w.r.t. index variable `i` and replace the index expression (in the
  /// enclosing expression) with a workspace access expression.  The index
  /// variable `i` is retained in the enclosing expression and used to access
  /// the workspace, while `iw` replaces `i` in the index expression that
  /// computes workspace results.
  void workspace(IndexVar i, IndexVar iw, Format format, std::string name="");

  /// Store the index expression's result to the given workspace w.r.t. index
  /// variable `i` and replace the index expression (in the enclosing
  /// expression) with a workspace access expression.  The index variable `i` is
  /// retained in the enclosing expression and used to access the workspace,
  /// while `iw` replaces `i` in the index expression that computes workspace
  /// results.
  void workspace(IndexVar i, IndexVar iw, TensorVar workspace);

  /// Returns the schedule of the index expression.
  const Schedule& getSchedule() const;

  /// Casts index expression to specified subtype.
  template <typename SubType>
  SubType as() {
    return to<SubType>(*this);
  }

  /// Visit the index expression's sub-expressions.
  void accept(IndexExprVisitorStrict *) const;

  /// Print the index expression.
  friend std::ostream& operator<<(std::ostream&, const IndexExpr&);
};

/// Check if two index expressions are isomorphic.
bool isomorphic(IndexExpr, IndexExpr);

/// Compare two index expressions by value.
bool equals(IndexExpr, IndexExpr);

/// Construct and returns an expression that negates this expression.
/// ```
/// A(i,j) = -B(i,j);
/// ```
IndexExpr operator-(const IndexExpr&);

/// Add two index expressions.
/// ```
/// A(i,j) = B(i,j) + C(i,j);
/// ```
IndexExpr operator+(const IndexExpr&, const IndexExpr&);

/// Subtract an index expressions from another.
/// ```
/// A(i,j) = B(i,j) - C(i,j);
/// ```
IndexExpr operator-(const IndexExpr&, const IndexExpr&);

/// Multiply two index expressions.
/// ```
/// A(i,j) = B(i,j) * C(i,j);  // Component-wise multiplication
/// ```
IndexExpr operator*(const IndexExpr&, const IndexExpr&);

/// Divide an index expression by another.
/// ```
/// A(i,j) = B(i,j) / C(i,j);  // Component-wise division
/// ```
IndexExpr operator/(const IndexExpr&, const IndexExpr&);


/// An index expression that represents a tensor access, such as `A(i,j))`.
/// Access expressions are returned when calling the overloaded operator() on
/// a `TensorVar`.  Access expressions can also be assigned an expression, which
/// happens when they occur on the left-hand-side of an assignment.
///
/// @see TensorVar Calling `operator()` on a `TensorVar` returns an `Assign`.
class Access : public IndexExpr {
public:
  Access() = default;
  Access(const Access&) = default;
  Access(const AccessNode*);
  Access(const TensorVar& tensorVar, const std::vector<IndexVar>& indices={},
         const std::map<int, std::shared_ptr<IndexVarIterationModifier>>& modifiers={},
         bool isAccessingStructure=false);

  /// Return the Access expression's TensorVar.
  const TensorVar &getTensorVar() const;

  /// Returns the index variables used to index into the Access's TensorVar.
  const std::vector<IndexVar>& getIndexVars() const;

  /// Returns whether access expression returns sparsity pattern of tensor.
  /// If true, the access expression returns 1 for every physically stored 
  /// component. If false, the access expression returns the value that is  
  /// stored for each corresponding component.
  bool isAccessingStructure() const;

  /// hasWindowedModes returns true if any accessed modes are windowed.
  bool hasWindowedModes() const;

  /// Returns whether or not the input mode (0-indexed) is windowed.
  bool isModeWindowed(int mode) const;

  /// Return the {lower,upper} bound of the window on the input mode (0-indexed).
  int getWindowLowerBound(int mode) const;
  int getWindowUpperBound(int mode) const;

  /// getWindowSize returns the dimension size of a window.
  int getWindowSize(int mode) const;

  /// getStride returns the stride of a window.
  int getStride(int mode) const;

  /// hasIndexSetModes returns true if any accessed modes have an index set.
  bool hasIndexSetModes() const;

  /// Returns whether or not the input mode (0-indexed) has an index set.
  bool isModeIndexSet(int mode) const;

  /// getModeIndexSetTensor returns a TensorVar corresponding to the Tensor that
  /// backs the index set for the input mode.
  TensorVar getModeIndexSetTensor(int mode) const;

  /// getIndexSet returns the index set of the input mode.
  const std::vector<int>& getIndexSet(int mode) const;

  /// Assign the result of an expression to a left-hand-side tensor access.
  /// ```
  /// a(i) = b(i) * c(i);
  /// ```
  Assignment operator=(const IndexExpr&);

  /// Must override the default Access operator=, otherwise it is a copy.
  Assignment operator=(const Access&);

  /// Must disambiguate TensorVar as it can be implicitly converted to IndexExpr
  /// and AccesExpr.
  Assignment operator=(const TensorVar&);

  /// Accumulate the result of an expression to a left-hand-side tensor access.
  /// ```
  /// a(i) += B(i,j) * c(j);
  /// ```
  Assignment operator+=(const IndexExpr&);

  typedef AccessNode Node;

  // Equality and comparison are overridden on Access to perform a deep
  // comparison of the access rather than a pointer check.
  friend bool operator==(const Access& a, const Access& b);
  friend bool operator<(const Access& a, const Access &b);
};


/// A literal index expression is a scalar literal that is embedded in the code.
/// @note In the future we may allow general tensor literals.
class Literal : public IndexExpr {
public:
  Literal() = default;
  Literal(const LiteralNode*);

  Literal(bool);
  Literal(unsigned char);
  Literal(unsigned short);
  Literal(unsigned int);
  Literal(unsigned long);
  Literal(unsigned long long);
  Literal(char);
  Literal(short);
  Literal(int);
  Literal(long);
  Literal(long long);
  Literal(int8_t);
  Literal(float);
  Literal(double);
  Literal(std::complex<float>);
  Literal(std::complex<double>);

  static Literal zero(Datatype);

  /// Returns the literal value.
  template <typename T> T getVal() const;

  /// Returns an untyped pointer to the literal value
  void* getValPtr();

  typedef LiteralNode Node;
};


/// A neg expression computes negates a number.
/// ```
/// a(i) = -b(i);
/// ```
class Neg : public IndexExpr {
public:
  Neg() = default;
  Neg(const NegNode*);
  Neg(IndexExpr a);

  IndexExpr getA() const;

  typedef NegNode Node;
};


/// An add expression adds two numbers.
/// ```
/// a(i) = b(i) + c(i);
/// ```
class Add : public IndexExpr {
public:
  Add();
  Add(const AddNode*);
  Add(IndexExpr a, IndexExpr b);

  IndexExpr getA() const;
  IndexExpr getB() const;

  typedef AddNode Node;
};


/// A sub expression subtracts two numbers.
/// ```
/// a(i) = b(i) - c(i);
/// ```
class Sub : public IndexExpr {
public:
  Sub();
  Sub(const SubNode*);
  Sub(IndexExpr a, IndexExpr b);

  IndexExpr getA() const;
  IndexExpr getB() const;

  typedef SubNode Node;
};


/// An mull expression multiplies two numbers.
/// ```
/// a(i) = b(i) * c(i);
/// ```
class Mul : public IndexExpr {
public:
  Mul();
  Mul(const MulNode*);
  Mul(IndexExpr a, IndexExpr b);

  IndexExpr getA() const;
  IndexExpr getB() const;

  typedef MulNode Node;
};


/// An div expression divides two numbers.
/// ```
/// a(i) = b(i) / c(i);
/// ```
class Div : public IndexExpr {
public:
  Div();
  Div(const DivNode*);
  Div(IndexExpr a, IndexExpr b);

  IndexExpr getA() const;
  IndexExpr getB() const;

  typedef DivNode Node;
};


/// A sqrt expression computes the square root of a number
/// ```
/// a(i) = sqrt(b(i));
/// ```
class Sqrt : public IndexExpr {
public:
  Sqrt() = default;
  Sqrt(const SqrtNode*);
  Sqrt(IndexExpr a);

  IndexExpr getA() const;

  typedef SqrtNode Node;
};


/// A cast expression casts a value to a specified type
/// ```
/// a(i) = cast<float>(b(i))
/// ```
class Cast : public IndexExpr {
public:
  Cast() = default;
  Cast(const CastNode*);
  Cast(IndexExpr a, Datatype newType);

  IndexExpr getA() const;

  typedef CastNode Node;
};

/// A call to an operator
class Call: public IndexExpr {
public:
  Call() = default;
  Call(const CallNode*);
  Call(const CallNode*, std::string name);

  const std::vector<IndexExpr>& getArgs() const;
  const std::function<ir::Expr(const std::vector<ir::Expr>&)> getFunc() const;
  const IterationAlgebra& getAlgebra() const;
  const std::vector<Property>& getProperties() const;
  const std::string getName() const;
  const std::map<std::vector<int>, std::function<ir::Expr(const std::vector<ir::Expr>&)>> getDefs() const;
  const std::vector<int>& getDefinedArgs() const;

  typedef CallNode Node;

private:
  std::string name;
};

/// A call to an intrinsic.
/// ```
/// a(i) = abs(b(i));
/// a(i) = pow(b(i),2);
/// ...
/// ```
class CallIntrinsic : public IndexExpr {
public:
  CallIntrinsic() = default;
  CallIntrinsic(const CallIntrinsicNode*);
  CallIntrinsic(const std::shared_ptr<Intrinsic>& func,
                const std::vector<IndexExpr>& args);

  const Intrinsic& getFunc() const;
  const std::vector<IndexExpr>& getArgs() const;

  typedef CallIntrinsicNode Node;
};

std::ostream& operator<<(std::ostream&, const IndexVar&);

/// Create calls to various intrinsics.
IndexExpr mod(IndexExpr, IndexExpr);
IndexExpr abs(IndexExpr);
IndexExpr pow(IndexExpr, IndexExpr);
IndexExpr square(IndexExpr);
IndexExpr cube(IndexExpr);
IndexExpr sqrt(IndexExpr);
IndexExpr cbrt(IndexExpr);
IndexExpr exp(IndexExpr);
IndexExpr log(IndexExpr);
IndexExpr log10(IndexExpr);
IndexExpr sin(IndexExpr);
IndexExpr cos(IndexExpr);
IndexExpr tan(IndexExpr);
IndexExpr asin(IndexExpr);
IndexExpr acos(IndexExpr);
IndexExpr atan(IndexExpr);
IndexExpr atan2(IndexExpr, IndexExpr);
IndexExpr sinh(IndexExpr);
IndexExpr cosh(IndexExpr);
IndexExpr tanh(IndexExpr);
IndexExpr asinh(IndexExpr);
IndexExpr acosh(IndexExpr);
IndexExpr atanh(IndexExpr);
IndexExpr gt(IndexExpr, IndexExpr);
IndexExpr lt(IndexExpr, IndexExpr);
IndexExpr gte(IndexExpr, IndexExpr);
IndexExpr lte(IndexExpr, IndexExpr);
IndexExpr eq(IndexExpr, IndexExpr);
IndexExpr neq(IndexExpr, IndexExpr);
IndexExpr max(IndexExpr, IndexExpr);
IndexExpr min(IndexExpr, IndexExpr);
IndexExpr heaviside(IndexExpr, IndexExpr = IndexExpr());

IndexExpr Not(IndexExpr);


/// A reduction over the components indexed by the reduction variable.
class Reduction : public IndexExpr {
public:
  Reduction() = default;
  Reduction(const ReductionNode*);
  Reduction(IndexExpr op, IndexVar var, IndexExpr expr);

  IndexExpr getOp() const;
  IndexVar getVar() const;
  IndexExpr getExpr() const;

  typedef ReductionNode Node;
};

/// Create a summation index expression.
Reduction sum(IndexVar i, IndexExpr expr);

/// Return true if the index statement is of the given subtype.  The subtypes
/// are Assignment, Forall, Where, Multi, and Sequence.
template <typename SubType> bool isa(IndexStmt);

/// Casts the index statement to the given subtype. Assumes S is a subtype and
/// the subtypes are Assignment, Forall, Where, Multi, and Sequence.
template <typename SubType> SubType to(IndexStmt);

/// A an index statement computes a tensor.  The index statements are
/// assignment, forall, where, multi, and sequence.
class IndexStmt : public util::IntrusivePtr<const IndexStmtNode> {
public:
  IndexStmt();
  IndexStmt(const IndexStmtNode* n);

  /// Visit the tensor expression
  void accept(IndexStmtVisitorStrict *) const;

  /// Return the free and reduction index variables in the assignment.
  std::vector<IndexVar> getIndexVars() const;

  /// Returns the domains/dimensions of the index variables in the statement.
  /// These are inferred from the dimensions they access.
  std::map<IndexVar,Dimension> getIndexVarDomains() const;

  /// Takes any index notation and concretizes unknowns to make it concrete notation
  IndexStmt concretize() const;

  /// Takes any index notation and concretizes unknowns to make it concrete notation
  /// given a Provenance Graph of indexVars
  IndexStmt concretizeScheduled(ProvenanceGraph provGraph, std::vector<IndexVar> forallIndexVarList) const;

  /// The \code{split} transformation splits (strip-mines) an index
  /// variable into two nested index variables, where the size of the
  /// inner index variable is constant.  The size of the outer index
  /// variable is the size of the original index variable divided by the
  /// size of the inner index variable, and the product of the new index
  /// variables sizes therefore equals the size of the original index
  /// variable.  Note that in the generated code, when the size of the
  /// inner index variable does not perfectly divide the original index
  /// variable, a \textit{tail strategy} is employed such as emitting a variable
  /// sized loop that handles remaining iterations.
  /// Preconditions: splitFactor is a positive nonzero integer
  IndexStmt split(IndexVar i, IndexVar i1, IndexVar i2, size_t splitFactor) const; // TODO: TailStrategy

  /// The divide transformation splits one index variable into
  /// two nested index variables, where the size of the outer
  /// index variable is constant.  The size of the inner index variable
  /// is thus the size of the original index variable divided by the
  /// size of the outer index variable.  The divide
  /// transformation is important in sparse codes because locating the
  /// starting point of a tile can require an $O(n)$ or $O(\log (n))$
  /// search.  Therefore, if we want to parallelize a blocked
  /// loop, then we want a fixed number of blocks and not a number
  /// proportional to the tensor size.
  /// Preconditions: divideFactor is a positive nonzero integer
  IndexStmt divide(IndexVar i, IndexVar i1, IndexVar i2, size_t divideFactor) const; // TODO: TailStrategy


  /// The reorder transformation swaps two directly nested index
  /// variables in an iteration graph.  This changes the order of
  /// iteration through the space and the order of tensor accesses.
  ///
  /// Preconditions:
  /// The precondition of a reorder transformation is that it must not hoist
  /// a tensor operation outside a reduction that it does not distribute
  /// over. Otherwise, this will alter the contents of a reduction and change the
  /// value of the result. In addition, we check that the result of the reorder
  /// transformation does not cause for tensors to be iterated out of order.
  /// Certain sparse data formats can only be accessed in a given mode ordering
  /// and we verify that this ordering is preserved after the reorder.
  IndexStmt reorder(IndexVar i, IndexVar j) const;

  /// reorder takes a new ordering for a set of index variables that are directly nested in the iteration order
  IndexStmt reorder(std::vector<IndexVar> reorderedvars) const;

  /// The mergeby transformation specifies how to merge iterators on
  /// the given index variable. By default, if an iterator is used for windowing
  /// it will be merged with the "gallop" strategy.
  /// All other iterators are merged with the "two finger" strategy.
  /// The two finger strategy merges by advancing each iterator one at a time, 
  /// while the gallop strategy implements the exponential search algorithm.
  /// 
  /// Preconditions:
  /// This command applies to variables involving sparse iterators only;
  /// it is a no-op if the variable invovles any dense iterators.
  /// Any variable can be merged with the two finger strategy, whereas gallop
  /// only applies to a variable if its merge lattice has a single point 
  /// (i.e. an intersection). For example, if a variable involves multiplications
  /// only, it can be merged with gallop.
  /// Furthermore, all iterators must be ordered for gallop to apply.
  IndexStmt mergeby(IndexVar i, MergeStrategy strategy) const;

  /// The parallelize
  /// transformation tags an index variable for parallel execution.  The
  /// transformation takes as an argument the type of parallel hardware
  /// to execute on.  The set of parallel hardware is extensible and our
  /// current code generation algorithm supports SIMD vector units, CPU
  /// threads, GPU thread blocks, GPU warps, and individual GPU threads.
  /// Parallelizing the iteration over an index variable changes the iteration
  /// order of the loop, and therefore requires reductions inside the
  /// iteration space described by the index variable's sub-tree in the
  /// iteration graph to be associative.  Furthermore, if the
  /// computation uses a reduction strategy that does not preserve the
  /// order, such as atomic instructions, then the reductions must also
  /// be commutative.
  ///
  /// Preconditions:
  /// Once a parallelize transformation is used, no other transformations may be
  /// applied on the iteration graph as the preconditions for other transformations assume
  /// serial code. In addition there are sometimes hardware-specific rules to how things can
  /// be parallelized such as a CUDA warp is a fixed size of 32 threads or to parallelize over
  /// CUDA threads then you must also parallelize over CUDA thread-blocks. These hardware-specific
  /// rules are checked in the code generator rather than before the transformation.
  ///
  /// In addition to hardware-specific preconditions, there are preconditions related to
  /// coiteration that apply for all hardware. An index variable that indexes
  /// into multiple sparse data structures cannot be parallelized as it is a while loop. Instead
  /// this loop can be parallelized by first strip-mining it with the split or divide
  /// transformation to create a parallel for loop with a serial nested while loop. Expressions
  /// that have an output in a format that does not support random insert can also not be
  /// parallelized. Parallelizing these expressions would require creating multiple copies of a
  /// datastructure and then merging them, which is left to future work. Note that there is a special
  /// case where the output's sparsity pattern is the same as one of the inputs.
  /// This true of the popular sampled dense-dense matrix multiply (SDDMM),
  /// tensor times vector (TTV), and tensor times matrix (TTM) kernels for example.
  /// This does not require creating multiple copies, but the precondition still
  /// prevents it as the implementation does not yet handle this special case.
  ///
  /// Finally, there are preconditions related to data races during reductions. The parallelize
  /// transformation allows for supplying a strategy to handle these data races. The NoRaces
  /// strategy has the precondition that there can be no reductions in the computation.
  /// The IgnoreRaces strategy has the precondition that for the given inputs the code generator can
  /// assume that no data races will occur. For all other strategies other than Atomics,
  /// there is the precondition
  /// that the racing reduction must be over the index variable being parallelized.
  IndexStmt parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy) const;

  /// pos and coord create
  /// new index variables in their respective iteration spaces.
  /// pos requires a tensor access expression as input, that
  /// describes the tensor whose coordinate hierarchy to perform a
  /// position cut with respect to.  Specifically, the derived ipos
  /// variable will iterate over the tensor's position space at the
  /// level that the i variable is used in the access expression
  ///
  /// Preconditions:
  /// The index variable supplied to the coord transformation must be in
  /// position space. The index variable supplied to the pos transformation must 
  /// be in coordinate space. The pos transformation also takes an input to
  /// indicate which position space to use. This input must appear in the computation
  /// expression and also be indexed by this index variable. In the case that this
  /// index variable is derived from multiple index variables, these variables must appear
  /// directly nested in the mode ordering of this datastructure. This allows for
  /// working with multi-dimensional position spaces.
  IndexStmt pos(IndexVar i, IndexVar ipos, Access access) const;
  // TODO: coord

  /// The fuse transformation collapses two directly nested index
  /// variables.  It results in a new fused index variable that iterates
  /// over the product of the coordinates of the fused index variables.
  /// This transformation by itself does not change iteration order, but
  /// facilitates other transformations such as iterating over the
  /// position space of several variables and distributing a
  /// multi-dimensional loop nest across a thread array on GPUs.
  ///
  /// Preconditions:
  /// The fuse transformation takes in two index variables. The second
  /// index variable must be directly nested under the first index variable in
  /// the iteration graph. In addition, the first index variable must be in
  /// coordinate space. To work with a multi-dimensional position space,
  /// it is instead necessary to fuse the coordinate dimensions and then use the
  /// pos transformation. This allows us to isolate the necessary preconditions
  /// to the pos transformation.
  IndexStmt fuse(IndexVar i, IndexVar j, IndexVar f) const;

  /// The precompute transformation is described in kjolstad2019
  /// allows us to leverage scratchpad memories and
  /// reorder computations to increase locality
  IndexStmt precompute(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace) const;

  ///  The precompute transformation is described in kjolstad2019
  ///  allows us to leverage scratchpad memories and
  ///  reorder computations to increase locality
  IndexStmt precompute(IndexExpr expr, std::vector<IndexVar> i_vars,
                       std::vector<IndexVar> iw_vars, TensorVar workspace) const;
  
  /// bound specifies a compile-time constraint on an index variable's
  /// iteration space that allows knowledge of the
  /// size or structured sparsity pattern of the inputs to be
  /// incorporated during bounds propagation
  ///
  /// Preconditions:
  /// The precondition for bound is that the computation bounds supplied are 
  /// correct given the inputs that this code will be run on.
  IndexStmt bound(IndexVar i, IndexVar i1, size_t bound, BoundType bound_type) const;

  /// The unroll primitive unrolls the corresponding loop by a statically-known
  /// integer number of iterations
  /// Preconditions: unrollFactor is a positive nonzero integer
  IndexStmt unroll(IndexVar i, size_t unrollFactor) const;

  /// The assemble primitive specifies whether a result tensor should be 
  /// assembled by appending or inserting nonzeros into the result tensor.
  /// In the latter case, the transformation inserts additional loops to 
  /// precompute statistics about the result tensor that are required for 
  /// preallocating memory and coordinating insertions of nonzeros.
  IndexStmt assemble(TensorVar result, AssembleStrategy strategy, 
                     bool separately_schedulable = false) const;

  /// The wsaccel primitive specifies the dimensions of a workspace that will be accelerated.
  /// Acceleration means adding compressed acceleration datastructures (bitmap, coordinate list) to a dense workspace.
  /// shouldAccel controls whether acceleration will be applied.
  /// When shouldAccel is true, if accelIndexVars is empty, then all dimensions should be accelerated.
  /// When shouldAccel is true, if accelIndexVars is not empty, then dimensions in accelIndexVars will be accelerated.
  /// When shouldAccel is false, accelIndexVars is ignored.
  /// Currently, it only supports one-dimension acceleration. Acceleration is used by default.
  ///
  /// Precondition:
  /// Workspace can be accessed by the IndexVars in the accelIndexVars.
  IndexStmt wsaccel(TensorVar& ws, bool shouldAccel = true,const std::vector<IndexVar>& accelIndexVars ={});

  /// Casts index statement to specified subtype.
  template <typename SubType>
  SubType as() {
    return to<SubType>(*this);
  }
};

/// Check if two index statements are isomorphic.
bool isomorphic(IndexStmt, IndexStmt);

/// Compare two index statments by value.
bool equals(IndexStmt, IndexStmt);

/// Print the index statement.
std::ostream& operator<<(std::ostream&, const IndexStmt&);


/// An assignment statement assigns an index expression to the locations in a
/// tensor given by an lhs access expression.
class Assignment : public IndexStmt {
public:
  Assignment() = default;
  Assignment(const AssignmentNode*);

  /// Create an assignment. Can specify an optional operator `op` that turns the
  /// assignment into a compound assignment, e.g. `+=`.
  Assignment(Access lhs, IndexExpr rhs, IndexExpr op = IndexExpr());

  /// Create an assignment. Can specify an optional operator `op` that turns the
  /// assignment into a compound assignment, e.g. `+=`. Additionally, specify
  /// any modifers on reduction index variables (windows, index sets, etc.).
  Assignment(TensorVar tensor, std::vector<IndexVar> indices, IndexExpr rhs,
             IndexExpr op = IndexExpr(),
             const std::map<int, std::shared_ptr<IndexVarIterationModifier>>& modifiers = {});

  /// Return the assignment's left-hand side.
  Access getLhs() const;

  /// Return the assignment's right-hand side.
  IndexExpr getRhs() const;

  /// Return the assignment compound operator (e.g., `+=`) or an undefined
  /// expression if the assignment is not compound (`=`).
  IndexExpr getOperator() const;

  /// Return the free index variables in the assignment, which are those used to
  /// access the left-hand side.
  const std::vector<IndexVar>& getFreeVars() const;

  /// Return the reduction index variables i nthe assign
  std::vector<IndexVar> getReductionVars() const;

  /// Return the set relation of indexVars in lhs and rhs
  IndexSetRel getIndexSetRel() const;

  typedef AssignmentNode Node;
};


class Yield : public IndexStmt {
public:
  Yield() = default;
  Yield(const YieldNode*);

  Yield(const std::vector<IndexVar>& indexVars, IndexExpr expr);

  const std::vector<IndexVar>& getIndexVars() const;

  IndexExpr getExpr() const;

  typedef YieldNode Node;
};


/// A forall statement binds an index variable to values and evaluates the
/// sub-statement for each of these values.
class Forall : public IndexStmt {
public:
  Forall() = default;
  Forall(const ForallNode*);
  Forall(IndexVar indexVar, IndexStmt stmt);
  Forall(IndexVar indexVar, IndexStmt stmt, MergeStrategy merge_strategy, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, size_t unrollFactor = 0);

  IndexVar getIndexVar() const;
  IndexStmt getStmt() const;

  ParallelUnit getParallelUnit() const;
  OutputRaceStrategy getOutputRaceStrategy() const;
  MergeStrategy getMergeStrategy() const;

  size_t getUnrollFactor() const;

  typedef ForallNode Node;
};

/// Create a forall index statement.
Forall forall(IndexVar i, IndexStmt stmt);
Forall forall(IndexVar i, IndexStmt stmt, MergeStrategy merge_strategy, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, size_t unrollFactor = 0);


/// A where statment has a producer statement that binds a tensor variable in
/// the environment of a consumer statement.
class Where : public IndexStmt {
public:
  Where() = default;
  Where(const WhereNode*);
  Where(IndexStmt consumer, IndexStmt producer);

  IndexStmt getConsumer();
  IndexStmt getProducer();

  /**
   * Retrieve the result of this where statement;
   */
   TensorVar getResult();

  /**
   * Retrieve the temporary variable of this where statement.
   */
  TensorVar getTemporary();

  typedef WhereNode Node;
};

/// Create a where index statement.
Where where(IndexStmt consumer, IndexStmt producer);


/// A sequence statement has two statements, a definition and a mutation, that
/// are executed in sequence.  The defintion creates an index variable and the
/// mutation updates it.
class Sequence : public IndexStmt {
public:
  Sequence() = default;
  Sequence(const SequenceNode*);
  Sequence(IndexStmt definition, IndexStmt mutation);

  IndexStmt getDefinition() const;
  IndexStmt getMutation() const;

  typedef SequenceNode Node;
};

/// Create a sequence index statement.
Sequence sequence(IndexStmt definition, IndexStmt mutation);


class Assemble : public IndexStmt {
public:
  typedef std::map<TensorVar,std::vector<std::vector<TensorVar>>> AttrQueryResults;

  Assemble() = default;
  Assemble(const AssembleNode*);
  Assemble(IndexStmt queries, IndexStmt compute, AttrQueryResults results);

  IndexStmt getQueries() const;
  IndexStmt getCompute() const;

  const AttrQueryResults& getAttrQueryResults() const;

  typedef AssembleNode Node;
};

/// Create an assemble index statement.
Assemble assemble(IndexStmt queries, IndexStmt compute, 
                  Assemble::AttrQueryResults results);


/// A multi statement has two statements that are executed separately, and let
/// us compute more than one tensor in a concrete index notation statement.
class Multi : public IndexStmt {
public:
  Multi() = default;
  Multi(const MultiNode*);
  Multi(IndexStmt stmt1, IndexStmt stmt2);

  IndexStmt getStmt1() const;
  IndexStmt getStmt2() const;

  typedef MultiNode Node;
};

/// Create a multi index statement.
Multi multi(IndexStmt stmt1, IndexStmt stmt2);

/// IndexVarInterface is a marker superclass for IndexVar-like objects.
/// It is intended to be used in situations where many IndexVar-like objects
/// must be stored together, like when building an Access AST node where some
/// of the access variables are windowed. Use cases for IndexVarInterface
/// will inspect the underlying type of the IndexVarInterface. For sake of
/// completeness, the current implementers of IndexVarInterface are:
/// * IndexVar
/// * WindowedIndexVar
/// * IndexSetVar
/// If this set changes, make sure to update the match function.
class IndexVarInterface {
public:
  virtual ~IndexVarInterface() = default;

  /// match performs a dynamic case analysis of the implementers of IndexVarInterface
  /// as a utility for handling the different values within. It mimics the dynamic
  /// type assertion of Go.
  static void match(
      std::shared_ptr<IndexVarInterface> ptr,
      std::function<void(std::shared_ptr<IndexVar>)> ivarFunc,
      std::function<void(std::shared_ptr<WindowedIndexVar>)> wvarFunc,
      std::function<void(std::shared_ptr<IndexSetVar>)> isetVarFunc
  ) {
    auto iptr = std::dynamic_pointer_cast<IndexVar>(ptr);
    auto wptr = std::dynamic_pointer_cast<WindowedIndexVar>(ptr);
    auto sptr = std::dynamic_pointer_cast<IndexSetVar>(ptr);
    if (iptr != nullptr) {
      ivarFunc(iptr);
    } else if (wptr != nullptr) {
      wvarFunc(wptr);
    } else if (sptr != nullptr) {
      isetVarFunc(sptr);
    } else {
      taco_iassert("IndexVarInterface was not IndexVar, WindowedIndexVar or IndexSetVar");
    }
  }
};

/// WindowedIndexVar represents an IndexVar that has been windowed. For example,
///   A(i) = B(i(2, 4))
/// In this case, i(2, 4) is a WindowedIndexVar. WindowedIndexVar is defined
/// before IndexVar so that IndexVar can return objects of type WindowedIndexVar.
class WindowedIndexVar : public util::Comparable<WindowedIndexVar>, public IndexVarInterface {
public:
  WindowedIndexVar(IndexVar base, int lo = -1, int hi = -1, int stride = 1);
  ~WindowedIndexVar() = default;

  /// getIndexVar returns the underlying IndexVar.
  IndexVar getIndexVar() const;

  /// get{Lower,Upper}Bound returns the {lower,upper} bound of the window of
  /// this index variable.
  int getLowerBound() const;
  int getUpperBound() const;
  /// getStride returns the stride to access the window by.
  int getStride() const;

  /// getWindowSize returns the number of elements in the window.
  int getWindowSize() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// IndexSetVar represents an IndexVar that has been projected via a set
/// of values. For example,
///  A(i) = B(i({1, 3, 5}))
/// projects the elements of B to be just elements at indexes 1, 3 and 5. In
/// this case, i({1, 3, 5}) is an IndexSetvar.
class IndexSetVar : public util::Comparable<IndexSetVar>, public IndexVarInterface {
public:
  IndexSetVar(IndexVar base, std::vector<int> indexSet);
  ~IndexSetVar() = default;

  /// getIndexVar returns the underlying IndexVar.
  IndexVar getIndexVar() const;
  /// getIndexSet returns the index set.
  const std::vector<int>& getIndexSet() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Index variables are used to index into tensors in index expressions, and
/// they represent iteration over the tensor modes they index into.
class IndexVar : public IndexExpr, public IndexVarInterface {

public:
  IndexVar();
  ~IndexVar() = default;
  IndexVar(const std::string& name);
  IndexVar(const std::string& name, const Datatype& type);
  IndexVar(const IndexVarNode *);

  /// Returns the name of the index variable.
  std::string getName() const;

  // Need these to overshadow the comparisons in for the IndexExpr instrusive pointer
  friend bool operator==(const IndexVar&, const IndexVar&);
  friend bool operator<(const IndexVar&, const IndexVar&);
  friend bool operator!=(const IndexVar&, const IndexVar&);
  friend bool operator>=(const IndexVar&, const IndexVar&);
  friend bool operator<=(const IndexVar&, const IndexVar&);
  friend bool operator>(const IndexVar&, const IndexVar&);

  typedef IndexVarNode Node;

  /// Indexing into an IndexVar returns a window into it.
  WindowedIndexVar operator()(int lo, int hi, int stride = 1);

  /// Indexing into an IndexVar with a vector returns an index set into it.
  IndexSetVar operator()(std::vector<int>&& indexSet);
  IndexSetVar operator()(std::vector<int>& indexSet);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

struct IndexVar::Content {
  std::string name;
};

struct WindowedIndexVar::Content {
  IndexVar base;
  int lo;
  int hi;
  int stride;
};

struct IndexSetVar::Content {
  IndexVar base;
  std::vector<int> indexSet;
};

std::ostream& operator<<(std::ostream&, const std::shared_ptr<IndexVarInterface>&);
std::ostream& operator<<(std::ostream&, const IndexVar&);
std::ostream& operator<<(std::ostream&, const WindowedIndexVar&);
std::ostream& operator<<(std::ostream&, const IndexSetVar&);

/// A suchthat statement provides a set of IndexVarRel that constrain
/// the iteration space for the child concrete index notation
class SuchThat : public IndexStmt {
public:
  SuchThat() = default;
  SuchThat(const SuchThatNode*);
  SuchThat(IndexStmt stmt, std::vector<IndexVarRel> predicate);

  IndexStmt getStmt() const;
  std::vector<IndexVarRel> getPredicate() const;

  typedef SuchThatNode Node;
};

/// Create a suchthat index statement.
SuchThat suchthat(IndexStmt stmt, std::vector<IndexVarRel> predicate);

/// A tensor variable in an index expression, which can either be an operand
/// or the result of the expression.
class TensorVar : public util::Comparable<TensorVar> {
public:
  TensorVar();
  TensorVar(const Type& type, const Literal& fill = Literal());
  TensorVar(const std::string& name, const Type& type, const Literal& fill = Literal());
  TensorVar(const Type& type, const Format& format, const Literal& fill = Literal());
  TensorVar(const std::string& name, const Type& type, const Format& format, const Literal& fill = Literal());
  TensorVar(const int &id, const std::string& name, const Type& type, const Format& format,
            const Literal& fill = Literal());

  /// Returns the ID of the tensor variable.
  int getId() const;

  /// Returns the name of the tensor variable.
  std::string getName() const;

  /// Returns the order of the tensor (number of modes).
  int getOrder() const;

  /// Returns the type of the tensor variable.
  const Type& getType() const;

  /// Returns the format of the tensor variable.
  const Format& getFormat() const;

  /// Returns the schedule of the tensor var, which describes how to compile
  /// and execute it's expression.
  const Schedule& getSchedule() const;

  /// Gets the fill value of the tensor variable. May be left undefined.
  const Literal& getFill() const;

  /// Gets the acceleration dimensions
  const std::vector<IndexVar>& getAccelIndexVars() const;

  /// Gets the acceleration flag
  bool getShouldAccel() const;

  /// Set the acceleration dimensions
  void setAccelIndexVars(const std::vector<IndexVar>& accelIndexVars, bool shouldAccel);

  /// Set the fill value of the tensor variable
  void setFill(const Literal& fill);

  /// Set the name of the tensor variable.
  void setName(std::string name);

  /// Check whether the tensor variable is defined.
  bool defined() const;

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()(const std::vector<IndexVar>& indices) const;

  /// Create an index expression that accesses (reads) this tensor.
  template <typename... IndexVars>
  const Access operator()(const IndexVars&... indices) const {
    return static_cast<const TensorVar*>(this)->operator()({indices...});
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices);

  /// Create an index expression that accesses (reads or writes) this tensor.
  template <typename... IndexVars>
  Access operator()(const IndexVars&... indices) {
    return this->operator()({indices...});
  }

  /// Assign a scalar expression to a scalar tensor.
  Assignment operator=(IndexExpr);

  /// Add a scalar expression to a scalar tensor.
  Assignment operator+=(IndexExpr);

  friend bool operator==(const TensorVar&, const TensorVar&);
  friend bool operator<(const TensorVar&, const TensorVar&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorVar&);


/// Check whether the statment is in the einsum index notation dialect.
/// This means the statement is an assignment, does not have any reduction
/// nodes, and is a sum of product, e.g., `a*...*b + ... + c*...*d`.    You can
/// optionally pass in a pointer to a string that the reason why it is not
/// concrete notation is printed to.
bool isEinsumNotation(IndexStmt, std::string* reason=nullptr);

/// Check whether the statement is in the reduction index notation dialect.
/// This means the statement is an assignment and that every reduction variable
/// has a reduction node nested above all variable uses.  You can optionally
/// pass in a pointer to a string that the reason why it is not concrete
/// notation is printed to.
bool isReductionNotation(IndexStmt, std::string* reason=nullptr);

/// Check whether the statement is in the reduction index notation dialect
/// given a schedule described by the Provenance Graph
bool isReductionNotationScheduled(IndexStmt, ProvenanceGraph, std::string* reason=nullptr);

/// Check whether the statement is in the concrete index notation dialect.
/// This means every index variable has a forall node, each index variable used
/// for computation is under a forall node for that variable, there are no reduction
/// nodes, and that every reduction variable use is nested inside a compound
/// assignment statement.  You can optionally pass in a pointer to a string
/// that the reason why it is not concrete notation is printed to.
bool isConcreteNotation(IndexStmt, std::string* reason=nullptr);

/// Convert einsum notation to reduction notation, by applying Einstein's
/// summation convention to sum non-free/reduction variables over their term.
Assignment makeReductionNotation(Assignment);
IndexStmt makeReductionNotation(IndexStmt);

/// Convert reduction notation to concrete notation, by inserting forall nodes,
/// replacing reduction nodes by compound assignments, and inserting temporaries
/// as needed.
IndexStmt makeConcreteNotation(IndexStmt);


/// Convert einsum notation to reduction notation, by applying Einstein's
/// summation convention to sum non-free/reduction variables over their term
/// taking into account a schedule given by the Provenance Graph.
Assignment makeReductionNotationScheduled(Assignment, ProvenanceGraph);
IndexStmt makeReductionNotationScheduled(IndexStmt, ProvenanceGraph);

/// Convert reduction notation to concrete notation, by inserting forall nodes,
/// replacing reduction nodes by compound assignments, and inserting temporaries
/// as needed while taking into account a schedule given by the Provenance Graph.
IndexStmt makeConcreteNotationScheduled(IndexStmt, ProvenanceGraph, std::vector<IndexVar> forallIndexVars);

/// Returns the results of the index statement, in the order they appear.
std::vector<TensorVar> getResults(IndexStmt stmt);

/// Returns the input tensors to the index statement, in the order they appear.
std::vector<TensorVar> getArguments(IndexStmt stmt);

/// Returns true iff all of the loops over free variables come before all of the loops over
/// reduction variables. Therefore, this returns true if the reduction controlled by the loops
/// does not a scatter.
bool allForFreeLoopsBeforeAllReductionLoops(IndexStmt stmt);

  /// Returns the temporaries in the index statement, in the order they appear.
std::vector<TensorVar> getTemporaries(IndexStmt stmt);

/// Returns the attribute query results in the index statement, in the order 
/// they appear.
std::vector<TensorVar> getAttrQueryResults(IndexStmt stmt);

// [Olivia]
/// Returns the temporaries in the index statement, in the order they appear.
std::map<Forall, Where> getTemporaryLocations(IndexStmt stmt);

/// Returns the results in the index statement that should be assembled by 
/// ungrouped insertion.
std::vector<TensorVar> getAssembledByUngroupedInsertion(IndexStmt stmt);

/// Returns the tensors in the index statement.
std::vector<TensorVar> getTensorVars(IndexStmt stmt);

/// Returns the result accesses, in the order they appear, as well as the set of
/// result accesses that are reduced into.
std::pair<std::vector<Access>,std::set<Access>> getResultAccesses(IndexStmt stmt);

/// Returns the input accesses, in the order they appear.
std::vector<Access> getArgumentAccesses(IndexStmt stmt);

/// Returns the index variables in the index statement.
std::vector<IndexVar> getIndexVars(IndexStmt stmt);

/// Returns the index variables in the index expression.
std::vector<IndexVar> getIndexVars(IndexExpr expr);

/// Returns the reduction variables in the index statement.
std::vector<IndexVar> getReductionVars(IndexStmt stmt);

/// Convert index notation tensor variables to IR pointer variables.
std::vector<ir::Expr> createVars(const std::vector<TensorVar>& tensorVars,
                                 std::map<TensorVar, ir::Expr>* vars, 
                                 bool isParameter=false);

/// Convert index notation tensor variables in the index statement to IR 
/// pointer variables.
std::map<TensorVar,ir::Expr> createIRTensorVars(IndexStmt stmt);


/// Simplify an index expression by setting the zeroed Access expressions to
/// zero and then propagating and removing zeroes.
IndexExpr zero(IndexExpr, const std::set<Access>& zeroed);

/// Simplify an index expression by setting the zeroed Access expressions to
/// zero and then propagating and removing zeroes.
IndexStmt zero(IndexStmt, const std::set<Access>& zeroed);

/// Infers the fill value of the input expression by applying properties if possible. If unable
/// to successfully infer the fill value of the result, returns the empty IndexExpr
IndexExpr inferFill(IndexExpr);

/// Returns true if there are no forall nodes in the indexStmt. Used to check
/// if the last loop is being lowered.
bool hasNoForAlls(IndexStmt);

/// Create an `other` tensor with the given name and format,
/// and return tensor(indexVars) = other(indexVars) if otherIsOnRight,
/// and otherwise returns other(indexVars) = tensor(indexVars).
IndexStmt generatePackStmt(TensorVar tensor,
                           std::string otherName, Format otherFormat, 
                           std::vector<IndexVar> indexVars, bool otherIsOnRight);

/// Same as generatePackStmt, where otherFormat is COO.
IndexStmt generatePackCOOStmt(TensorVar tensor, 
                              std::vector<IndexVar> indexVars, bool otherIsOnRight);

}
#endif
