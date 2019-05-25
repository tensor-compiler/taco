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
  py::class_<taco::IndexVar>(m, "indexvar")
          .def(py::init<>())
          .def(py::init<const std::string&>())
          .def("name", &taco::IndexVar::getName)

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

          // .def("__mod__", &mod, py::is_operator());

}

static void defineIndexExpr(py::module &m){

  py::class_<IndexExprNode, PyIndexExprNode> pyNode(m, "IndexExprNode");
  pyNode
      .def(py::init<>())
      .def("accept", &IndexExprNode::accept);

  auto exprClass = py::class_<IndexExpr>(m, "IndexExpr", pyNode)
          .def(py::init<int64_t>())
          .def(py::init<double>())
          .def_property_readonly("datatype", &IndexExpr::getDataType)

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


  defineBinaryIndexExpr<Add>(m, "Add");
  defineBinaryIndexExpr<Sub>(m, "Sub");
  defineBinaryIndexExpr<Mul>(m, "Mul");
  defineBinaryIndexExpr<Div>(m, "Div");
  defineUnaryExpr<Sqrt>(m, "Sqrt");
  defineUnaryExpr<Neg>(m, "Neg");
}

static void defineAccess(py::module &m){

  py::class_<Access, IndexExpr>(m, "Access")
          .def(py::init<>())
          .def(py::init<TensorVar, std::vector<IndexVar>>(), py::arg("tensorVar"), py::arg("indices") = py::list())
          .def("tensor_var", &Access::getTensorVar)
          .def("index_vars", &Access::getIndexVars);
}


void defineIndexNotation(py::module &m){
  m.def("get_index_vars", &getIndexVars);
  defineIndexVar(m);
  defineIndexExpr(m);
  defineAccess(m);
  //m.def("mod", &mod);
  m.def("abs", &abs);
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
  //m.def("not", &not);
  m.def("gt", &gt);
  m.def("lt", &lt);
  m.def("gte", &gte);
  m.def("lte", &lte);
  m.def("eq", &eq);
  m.def("neq", &neq);
  m.def("max", &max);
  m.def("min", &min);
  m.def("heaviside", &heaviside);
  m.def("sum", &sum);
}

}}

