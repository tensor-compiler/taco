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
>>> t1[1, 1] = 100
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


Notes
-----
Construction index expressions in this way can largely be ignored since taco will implicitly convert python scalars to index
expressions.


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


  defineBinaryIndexExpr<Add>(m, "add");
  defineBinaryIndexExpr<Sub>(m, "sub");
  defineBinaryIndexExpr<Mul>(m, "mul");
  defineBinaryIndexExpr<Div>(m, "div");
  defineUnaryExpr<Neg>(m, "neg");
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
>>> t1, t2 = pt.tensor(5), pt.tensor(-2)
>>> 1


)");
  m.def("abs", &abs, R"(
abs(e1)

Return the element-wise absolute value of the index expression.

Parameters
-----------
e1: index_expression
    Input expression

Returns
--------


)");
  m.def("pow", &pow);
  m.def("square", &square);
  m.def("cube", &cube);
  m.def("sqrt", &sqrt);
  m.def("cube_root", &cbrt);
  m.def("exp", &exp);
  m.def("log", &log);
  m.def("log10", &log10);
  m.def("sin", &sin);
  m.def("cos", &cos);
  m.def("tan", &tan);
  m.def("asin", &asin);
  m.def("acos", &acos);
  m.def("atan", &atan);
  m.def("atan2", &atan2);
  m.def("sinh", &sinh);
  m.def("cosh", &cosh);
  m.def("tanh", &tanh);
  m.def("asinh", &asinh);
  m.def("acosh", &acosh);
  m.def("atanh", &atanh);
  m.def("logical_not", &Not);
  m.def("gt", &gt);
  m.def("lt", &lt);
  m.def("ge", &gte);
  m.def("le", &lte);
  m.def("eq", &eq);
  m.def("ne", &neq);
  m.def("max", &max);
  m.def("min", &min);
  m.def("heaviside", &heaviside);
  m.def("sum", &sum);
}

}}

