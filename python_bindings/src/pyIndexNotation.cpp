#include "pyIndexNotation.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes_abstract.h"
#include "taco/index_notation/index_notation_nodes.h"

PYBIND11_DECLARE_HOLDER_TYPE(T, taco::util::IntrusivePtr<T>, true)

namespace taco{
namespace pythonBindings{

class PyIndexExprNode : public IndexExprNode{
public:

  using IndexExprNode::IndexExprNode;

  void accept(IndexExprVisitorStrict *visitor) const override {
    PYBIND11_OVERLOAD_PURE(
            void,            /* Return type */
            IndexExprNode,   /* Parent class */
            accept,          /* Name of function in C++ (must match Python name) */
            visitor          /* Argument(s) */
    )
  }
};

static void defineIndexVar(py::module &m){
  py::options options;
  options.disable_function_signatures();

  py::class_<taco::IndexVar>(m, "index_var", R"(
index_var(name)

Creates an index variable that can be used to access a :class:`pytaco.tensor`.

Makes an index variable with the name, name to access a dimension of a given tensor.

Parameters
-----------
name: str, optional
    The name PyTaco assigns to the index_var created. If no name is specified, PyTaco generated its own name for the
    index_var.

Attributes
------------
name

Examples
-----------
Index variables must be declared before use. So for example, if we need to make an index expression that adds two
matrices we would write:

>>> import pytaco as pt
>>> i, j = pt.index_var(), pt.index_var()
>>> t1 = pt.tensor([2, 2])
>>> t2 = pt.tensor([2, 2])
>>> t1.insert([1, 1], 100)
>>> add_expr = t1[i, j] + t2[i, j]


This can get cumbersome if we need a large number of :class:`pytaco.index_var` s so PyTaco provides a convenience
function to return a list of :class:`pytaco.index_var` s called :func:`pytaco.get_index_vars`.

This means line 2 above could be replaced by:

>>> i, j = pt.get_index_vars(2)

Notes
-------
Index variables with the same name cannot be used to access different dimensions of the same tensor. This is a feature
that taco currently does not support.

)")
          .def(py::init<>())
          .def(py::init<const std::string&>())
          .def_property_readonly("name", &taco::IndexVar::getName, R"(
Returns the name of the :class:`pytaco.index_var`
)")

          .def("__eq__", [](const taco::IndexVar& self, const taco::IndexVar& other) -> bool{
              return self == other;
          }, py::is_operator())

          .def("__ne__", [](const taco::IndexVar& self, const taco::IndexVar& other) -> bool{
              return self != other;
          }, py::is_operator())

          .def("__lt__", [](const taco::IndexVar& self, const taco::IndexVar& other) -> bool{
              return self < other;
          }, py::is_operator())

          .def("__le__", [](const taco::IndexVar& self, const taco::IndexVar& other) -> bool{
              return self <= other;
          }, py::is_operator())

          .def("__ge__", [](const taco::IndexVar& self, const taco::IndexVar& other) -> bool{
              return self >= other;
          }, py::is_operator())

          .def("__gt__", [](const taco::IndexVar& self, const taco::IndexVar& other) -> bool{
              return self > other;
          }, py::is_operator())

          .def("__repr__", [](const taco::IndexVar& indexVar) -> std::string{
              std::ostringstream o;
              o << "IndexVar(" << indexVar << ")";
              return o.str();
          }, py::is_operator());
}

template<class T>
static void defineBinaryIndexExpr(py::module &m, const std::string& pyclassName){

  py::class_<T, IndexExpr>(m, pyclassName.c_str())
        .def(py::init<>())
        .def(py::init<IndexExpr, IndexExpr>())
        .def("get_a", &T::getA)
        .def("get_b", &T::getB)

        .def("__repr__", [](const T& expr) -> std::string{
            std::ostringstream o;
            o << "IndexExpr(" << expr << ")";
            return o.str();
        }, py::is_operator());

}

template<class T>
static void defineUnaryExpr(py::module &m, const std::string& pyclassName){

  py::class_<T, IndexExpr>(m, pyclassName.c_str())
          .def(py::init<>())
          .def(py::init<IndexExpr>())
          .def("get_a", &T::getA)

          .def("__repr__", [](const T& expr) -> std::string{
              std::ostringstream o;
              o << "IndexExpr(" << expr << ")";
              return o.str();
          }, py::is_operator());
}

static void defineCast(py::module &m){

  py::class_<Cast, IndexExpr>(m, "cast_c")
          .def(py::init<IndexExpr, Datatype>())
          .def("get_a", &Cast::getA)

          .def("__repr__", [](const Cast& expr) -> std::string{
              std::ostringstream o;
              o << "IndexExpr(" << expr << ")";
              return o.str();
          }, py::is_operator());
}

static std::vector<IndexVar> getIndexVars(int n){
  std::vector<IndexVar> vars;
  for(int i = 0; i < n; ++i){
    vars.emplace_back(IndexVar());
  }
  return vars;
}

static void defineReduction(py::module &m){

  py::class_<Reduction, IndexExpr>(m, "reduction")
          .def(py::init<>())
          .def(py::init<IndexExpr, IndexVar, IndexExpr>())
          .def("get_op", &Reduction::getOp)
          .def("get_var", &Reduction::getVar)
          .def("get_expr", &Reduction::getExpr)

          .def("__repr__", [](const Reduction& expr) -> std::string{
              std::ostringstream o;
              o << "IndexExpr(" << expr << ")";
              return o.str();
          }, py::is_operator());
}

template<typename PyClass>
static void addIndexExprOps(PyClass &class_instance){

  class_instance
          .def("__add__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
            return self + other;
          }, py::is_operator())

          .def("__radd__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              return other + self;
          }, py::is_operator())

          .def("__sub__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              return self - other;
          }, py::is_operator())

          .def("__rsub__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              return other - self;
          }, py::is_operator())

          .def("__mul__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              return self * other;
          }, py::is_operator())

          .def("__rmul__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              return other * self;
          }, py::is_operator())

          .def("__div__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              IndexExpr cast = new CastNode(self, Float64);
              return new DivNode(cast, other);
          }, py::is_operator())

          .def("__rdiv__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              IndexExpr cast = new CastNode(self, Float64);
              return new DivNode(other, cast);
          }, py::is_operator())

          .def("__truediv__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              IndexExpr cast = new CastNode(self, Float64);
              return new DivNode(cast, other);
          }, py::is_operator())

          .def("__rtruediv__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              IndexExpr cast = new CastNode(self, Float64);
              return new DivNode(other, cast);
          }, py::is_operator())

          .def("__floordiv__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              IndexExpr div = new DivNode(self, other);
              return new CastNode(div, Int64);
          }, py::is_operator())

          .def("__rfloordiv__", [](const IndexExpr &self, const IndexExpr &other) -> IndexExpr{
              IndexExpr div = new DivNode(other, self);
              return new CastNode(div, Int64);
          }, py::is_operator())

          .def("__pow__", [](const IndexExpr &self, const IndexExpr &other, py::object modulo) -> IndexExpr{
              if(!modulo.is_none()){
                throw py::value_error("Modulo not currently supported");
              }
              return pow(self, other);
          }, py::is_operator(), py::arg("other"), py::arg() = py::none())

          .def("__neg__", [](const IndexExpr &a) -> IndexExpr {
              return -a;
          }, py::is_operator())

          .def("__gt__", &gt, py::is_operator())

          .def("__lt__", &lt, py::is_operator())

          .def("__ge__", &gte, py::is_operator())

          .def("__le__", &lte, py::is_operator())

          .def("__eq__", &eq, py::is_operator())

          .def("__ne__", &neq, py::is_operator())

          .def("__abs__", &abs, py::is_operator());
}

static void defineIndexExpr(py::module &m){

  py::options options;
  options.disable_function_signatures();

  py::class_<IndexExprNode, PyIndexExprNode> pyNode(m, "IndexExprNode");
  pyNode
      .def(py::init<>())
      .def("accept", &IndexExprNode::accept);

  auto exprClass = py::class_<IndexExpr>(m, "index_expression", pyNode, R"(
index_expression(num)

Creates an Index Expression.

This direct invocation is only used to convert python ints and floats to index expressions. In all other cases, index
expression will be formed by accessing a tensor using :class:`~pytaco.index_var` s and different operations on that
access as seen in the :ref:`expr_funcs` section.

Note that in general, actually performing computations using index expressions require users to specify an output tensor
with the correct shape. Dimensions indexed by the same :class:`index_var` must have the same shape. As a result,
determining the output shape is easy once the expression has been written. See the examples section.

Parameters
-----------
num: int, float
    The scalar value to use to make an index expression.

Attributes
------------
datatype

Examples
---------
>>> import pytaco as pt
>>> pt.index_expression(3)
IndexExpr(3)

Implicit conversion

>>> i, j = pt.get_index_vars(2)
>>> t = pt.tensor([3,3])
>>> t[i,j] = 10 # All values set to 10 since 10 implied to be index expr

Scalar access

>>> s = pt.tensor(100)
>>> scalar_expr = s[None] * t[i,j]

An example of determining the output shape using matrix multiply:

>>> a = pt.tensor([4, 2])
>>> b = pt.tensor([2, 10])

We can represent matrix multiply as ``C[i, j] = A[i, k] * B[k, j]``. Since we have the representation of the computation
and we know that dimensions indexed by the same index variable must have the same shape, we can construct C by letting
its first dimension have size ``A.shape[0]`` since both dimensions are indexed by ``i`` and letting its second dimension
have size B.shape[1] since ``j`` indexes both of those dimensions. This, we would contiue the above as follows:

>>> c = pt.tensor([a.shape[0], b.shape[1]])

Then we could write
>>> i, j, k = pt.get_index_vars(3)
>>> c[i, j] = a[i, k] * b[k, j]

Notes
-----
Construction index expressions in this way can largely be ignored since taco will implicitly convert python scalars to index
expressions.

Creating index expressions from 0 order tensors must be done by indexing the 0 order tensor with ``None``. This tells
taco that there are no dimensions to access.

The :func:`evaluate` function allows users to represent index notation as a string using parenthesis instead of
square brackets and not requiring that scalars be indexed with ``None``. This function can infer the output dimension
of the tensor given the input string but does not currently support all of the index expression functions available.


)")
          .def(py::init<int64_t>())
          .def(py::init<double>())
          .def_property_readonly("datatype", &IndexExpr::getDataType, R"(
Returns the data type this expression will output after computation.
)")

          .def("equals", (bool (*) (IndexExpr, IndexExpr)) &equals)

          .def("__repr__", [](const IndexExpr &a) -> std::string {
              std::ostringstream o;
              if(a.defined()){
                o << "IndexExpr(" << a << ")";
              }else{
                o << a;
              }

              return o.str();
          }, py::is_operator());

  py::implicitly_convertible<int64_t, IndexExpr>();
  py::implicitly_convertible<double, IndexExpr>();
  addIndexExprOps(exprClass);


  defineBinaryIndexExpr<Add>(m, "add_c");
  defineBinaryIndexExpr<Sub>(m, "sub_c");
  defineBinaryIndexExpr<Mul>(m, "mul_c");
  defineBinaryIndexExpr<Div>(m, "div_c");
  defineUnaryExpr<Neg>(m, "neg_c");
  defineCast(m);

}

static void defineAccess(py::module &m){

  py::class_<Access, IndexExpr>(m, "Access")
          .def(py::init<>())
          .def(py::init<TensorVar, std::vector<IndexVar>>(), py::arg("tensorVar"), py::arg("indices") = py::list())
          .def("tensor_var", &Access::getTensorVar)
          .def("index_vars", &Access::getIndexVars);
}


void defineIndexNotation(py::module &m){
  py::options options;
  options.disable_function_signatures();

  m.def("get_index_vars", &getIndexVars, R"(
get_index_vars(vars)

Creates a list of the number of index variables specified.

Parameters
----------
vars: int
    The number of index variables to create.

See also
---------
:class:`~pytaco.index_var`


Returns
--------
index_variables: list
    A list containing vars :class:`pytaco.index_var` s.


Examples
---------
We can create an arbitrary number of index variables easily as follows:

>>> import pytaco as pt
>>> list_of_vars = pt.get_index_vars(10)
>>> len(list_of_vars)
10

We can also immediately unpack the vars as follows:

>>> i, j, k = pt.get_index_vars(3)


)");
  defineIndexVar(m);
  defineIndexExpr(m);
  defineAccess(m);
  defineReduction(m);


  m.def("remainder", &mod, R"(
remainder(e1, e2)

Return element wise remainder of division.

The sign of the result is always equivalent to the dividend.

Warnings
--------
This should not be confused with the python modulus operator.

This is equivalent to C's remainder.

Parameters
------------
e1: index_expression
    The dividend expression

e2: index_expression
    The divisor expression

Returns
---------
An expression representing the element-wise remainder of the input tensors.


Examples
----------
>>> import pytaco as pt
>>> rem = pt.remainder(5, 2)
>>> t = pt.tensor()
>>> t[None] = rem
>>> t[0]
1.0


)");
  m.def("abs", &abs, R"(
abs(e1)

Return the element-wise absolute value of the index expression.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Examples
----------
>>> import pytaco as pt
>>> t = pt.as_tensor([-2, 0, 1])
>>> i = pt.index_var()
>>> abs_expr = pt.abs(t[i])

We can then assign this description to a tensor to actually perform the computation

>>> res_t = pt.tensor([3])
>>> res_t[i] = abs_expr
>>> res_t.to_array()
array([2., 0., 1.], dtype=float32)

The above tells taco to compute the absolute value expression and store it in the tensor res_t keeping the dimension
since ``i`` is specified in both the right hand side and the left hand side of the expression.


Returns
---------
abs_exp: index_expression
    An index expression representing the element wise absolute value of its inputs.
)");

  m.def("square", &square, R"(
square(e1)

Return the element-wise square value of the index expression.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Examples
----------
>>> import pytaco as pt
>>> t = pt.as_tensor([-2, 2, 1])
>>> i = pt.index_var()
>>> sq_expr = pt.square(t[i])

We can then assign this description to a tensor to actually perform the computation.

The code below tells taco to compute the square of each value, sum over all those values and store it in the tensor
res_t. Since ``i`` appears on the right hand side of the expression but not on the left, taco will take the sum of the
values produced.

>>> res_t = pt.tensor()
>>> res_t[None] = sq_expr
>>> res_t.to_array()
array(9., dtype=float32)

Returns
---------
sq_exp: index_expression
    An index expression representing the element wise square of the input expression.
)");

  m.def("cube", &cube, R"(
cube(e1)

Return the element-wise cube value of the index expression.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Examples
----------
The code below tells taco to compute the cube of each value, sum over all the j indices and store the result in res_t.
Since ``j`` appears on the right hand side of the expression but not on the left, taco will take the sum of the
values over the dimension indexed by j.

>>> import pytaco as pt
>>> t = pt.as_tensor([[-2, 2, 1], [2, 3, 1]])
>>> i, j = pt.get_index_vars(2)
>>> res_t = pt.tensor([t.shape[0]])
>>> res_t[i] = pt.cube(t[i, j])
>>> res_t.to_array()
array([ 1., 36.], dtype=float32)

Returns
---------
cube_exp: index_expression
    An index expression representing the element wise cube of the input expression.
)");
  m.def("sqrt", &sqrt, R"(
sqrt(e1)

Return the element-wise sqrt of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the square root.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Examples
----------

>>> import pytaco as pt
>>> t = pt.as_tensor([4, 16])
>>> i = pt.index_var()
>>> res_t = pt.tensor([t.shape[0]])
>>> res_t[i] = pt.sqrt(pt.cast(t[i], pt.float32))
>>> res_t.to_array()
array([2., 4.], dtype=float32)

Returns
---------
sqrt_exp: index_expression
    An index expression representing the element wise square root of the input expression.
)");

  m.def("cube_root", &cbrt, R"(
cube_root(e1)

Return the element-wise cube root of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
cbrt_expr: index_expression
    An index expression representing the element wise cube root of the input expression.
)");

  m.def("exp", &exp, R"(
exp(e1)

Calculate the exponential of all elements in an index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Examples
---------
We show computing the standard softmax function as an example.

>>> import pytaco as pt
>>> t = pt.as_tensor([[4, 0.3], [2,  7]])
>>> t = pt.as_type(t, pt.float32)
>>> exp_sum = pt.tensor([t.shape[0]], pt.dense)

>>> i, j = pt.get_index_vars(2)
>>> exp_sum[i] = pt.exp(t[i, j]) # sum across the rows and exp
>>> soft_max_t = pt.tensor(t.shape, pt.dense)
>>> soft_max_t[i, j] = pt.exp(t[i, j]) / exp_sum[i] # divide each row by its sum
>>> print(soft_max_t.to_array())
[[0.975873   0.02412702]
 [0.00669285 0.9933072 ]]

Returns
---------
exp_expr: index_expression
    An index expression representing the element wise exponent of the input expression.
)");

  m.def("log", &log, R"(
log(e1)

Return the element-wise logarithm base e of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
log_expr: index_expression
    An index expression representing the element wise base e logarithm of the input expression.
)");

  m.def("log10", &log10, R"(
log10(e1)

Return the element-wise logarithm base 10 of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
log10_expr: index_expression
    An index expression representing the element wise base 10 logarithm of the input expression.
)");
  m.def("sin", &sin, R"(
sin(e1)

Return the element-wise sine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
sine_expr: index_expression
    An index expression representing the element wise sine of the input expression.
)");
  m.def("cos", &cos, R"(
cos(e1)

Return the element-wise cosine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
cosine_expr: index_expression
    An index expression representing the element wise cosine of the input expression.
)");

  m.def("tan", &tan, R"(
tan(e1)

Return the element-wise tangent of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
tangent_expr: index_expression
    An index expression representing the element wise tangent of the input expression.
)");
  m.def("asin", &asin, R"(
asin(e1)

Return the element-wise arcsine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
asin_expr: index_expression
    An index expression representing the element wise arcsine of the input expression.
)");
  m.def("acos", &acos, R"(
acos(e1)

Return the element-wise arccosine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
acos_expr: index_expression
    An index expression representing the element wise arccosine of the input expression.
)");
  m.def("atan", &atan, R"(
atan(e1)

Return the element-wise arc tangent of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
atan_expr: index_expression
    An index expression representing the element wise arc tangent of the input expression.
)");
  m.def("atan2", &atan2, R"(
atan2(e1)

Return the element-wise arc tangent of the index expression respecting quadrants.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
atan2_expr: index_expression
    An index expression representing the element wise arc tangent of the input expression respecting quadrants.
)");
  m.def("sinh", &sinh, R"(
sinh(e1)

Return the element-wise hyperbolic sine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
sinh_expr: index_expression
    An index expression representing the element wise hyperbolic sine of the input expression.
)");
  m.def("cosh", &cosh, R"(
cosh(e1)

Return the element-wise hyperbolic cosine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
cosh_expr: index_expression
    An index expression representing the element wise hyperbolic cosine of the input expression.
)");
  m.def("tanh", &tanh,  R"(
tanh(e1)

Return the element-wise hyperbolic tangent of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
tanh_expr: index_expression
    An index expression representing the element wise hyperbolic tangent of the input expression.
)");
  m.def("asinh", &asinh, R"(
asinh(e1)

Return the element-wise hyperbolic arcsine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
asinh_expr: index_expression
    An index expression representing the element wise hyperbolic arcsine of the input expression.
)");
  m.def("acosh", &acosh, R"(
acosh(e1)

Return the element-wise hyperbolic arccosine of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
acosh_expr: index_expression
    An index expression representing the element wise hyperbolic arccosine of the input expression.
)");
  m.def("atanh", &atanh, R"(
atanh(e1)

Return the element-wise hyperbolic arc tangent of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
atanh_expr: index_expression
    An index expression representing the element wise hyperbolic arc tangent of the input expression.
)");

  m.def("logical_not", &Not, R"(
logical_not(e1)

Return the element-wise logical not of the index expression.

The index expression must have a floating point type. If necessary, a user may :func:`cast` the input expression before
applying the function as shown in :func:`sqrt`.

This must be assigned to a tensor for the computation to be performed.

Parameters
-----------
e1: index_expression
    Input index expression

Returns
---------
logical_not_expr: index_expression
    An index expression representing the element wise logical not of the input expression.
)");

  m.def("pow", &pow, R"(
pow(e1, e2)

Computes e1**e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_raised_to_e2_expr: index_expression
    An index expression representing raising every element in e1 to a power specified by to corresponding element in e2.
)");
  m.def("gt", &gt, R"(
gt(e1, e2)

Computes e1 > e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_gt_e2: index_expression
    An index expression computing a true value everywhere e1 > e2 and a false value in all other locations.
)");
  m.def("lt", &lt, R"(
lt(e1, e2)

Computes e1 < e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_lt_e2: index_expression
    An index expression computing a true value everywhere e1 < e2 and a false value in all other locations.
)");
  m.def("ge", &gte, R"(
ge(e1, e2)

Computes e1 >= e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_ge_e2: index_expression
    An index expression computing a true value everywhere e1 >= e2 and a false value in all other locations.
)");
  m.def("le", &lte, R"(
le(e1, e2)

Computes e1 <= e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_le_e2: index_expression
    An index expression computing a true value everywhere e1 <= e2 and a false value in all other locations.
)");
  m.def("eq", &eq, R"(
eq(e1, e2)

Computes e1 == e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_eq_e2: index_expression
    An index expression computing a true value everywhere e1 == e2 and a false value in all other locations.
)");
  m.def("ne", &neq, R"(
ne(e1, e2)

Computes e1 != e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
e1_ne_e2: index_expression
    An index expression computing a true value everywhere e1 != e2 and a false value in all other locations.
)");
  m.def("max", &max, R"(
max(e1, e2)

Computes max(e1, e2) element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
max: index_expression
    An index expression which computes the element wise maximum of two tensors
)");
  m.def("min", &min, R"(
min(e1, e2)

Computes min(e1, e2) element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
min: index_expression
    An index expression which computes the element wise minimum of two tensors
)");
  m.def("heaviside", &heaviside, R"(
heaviside(e1, e2)

Computes element wise heaviside as described in :func:`tensor_heaviside`.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
hs: index_expression
    An index expression which computes the element wise heaviside step function of two tensors
)");
  m.def("sum", &sum, R"(
sum(var, e)

Sums a dimension of an index expression.

Sums all elements in the dimension of the index expression specified by var.

Parameters
------------
var: index_var
    An index var corresponding to the dimension to sum across in the input index expression e.

e: index_expression
    The index expression to sum.


Notes
-------
This is different from the function :func:`tensor_sum` since this function can only sum one index variable at a time.
The different expressions can be chained together to get behavior similar to :func:`tensor_sum`.


Examples
----------
We can use this to compute the magnitute of a vector without needed to specify a tensor to imply the reduction.

>>> import pytaco as pt
>>> t = pt.as_tensor([2.0, 3, 4, 5, 11])
>>> i = pt.index_var()
>>> res = pt.tensor()
>>> res[None] = pt.sqrt(pt.sum(i, pt.square(t[i])))
>>> res[0]
13.22875690460205

Returns
---------
reduced: index_expression
    An index_expression with the dimension specified by var summed out.

)");


  m.def("neg", [](IndexExpr e) -> IndexExpr{
      return Neg(e);
  }, R"(
neg(e)

Represents computing the element wise negation of the input expression.

Parameters
-----------
e: index_expression
    Input index expression

Returns
---------
neg: index_expression
    An index expression representing the element wise negation of the input expression.

)");

  m.def("cast", [](IndexExpr e, Datatype dt) -> IndexExpr{
    return Cast(e, dt);
  }, R"(
cast(e, dt)

Cast an expression from one datatype to another.

Parameters
------------
e: index_expression
    The index expression to cast

dt: datatype
    The type of the output index expression.

Returns
---------
casted: index_expression
    An index expression which has been cast to a new type dt.

)");

  m.def("add", [](IndexExpr e1, IndexExpr e2) -> IndexExpr{
      return Add(e1, e2);
  },R"(
add(e1, e2)

Computes e1 + e2 element-wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
sum: index_expression
    An index expression which computes the element wise addition of two tensors

)");

  m.def("sub", [](IndexExpr e1, IndexExpr e2) -> IndexExpr{
      return Sub(e1, e2);
  }, R"(
sub(e1, e2)

Computes e1 - e2 element wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
diff: index_expression
    An index expression which computes the element wise difference between two tensors

)");

  m.def("mul", [](IndexExpr e1, IndexExpr e2) -> IndexExpr{
      return Mul(e1, e2);
  },R"(
mul(e1, e2)

Computes e1 * e2 element wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
product: index_expression
    An index expression which computes the element wise product of two tensors

)");

  m.def("div", [](IndexExpr e1, IndexExpr e2) -> IndexExpr{
      return Div(e1, e2);
  }, R"(
div(e1, e2)

Computes e1 / e2 element wise.

Parameters
-----------
e1, e2: index_expressions
    Input index expressions

Returns
---------
quotient: index_expression
    An index expression which computes the element wise quotient of two tensors

)");

}

}}

