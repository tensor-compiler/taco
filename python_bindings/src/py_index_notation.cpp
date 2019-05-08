#include "py_index_notation.h"
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
  py::class_<taco::IndexVar>(m, "indexVar")
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

static void defineReduction(py::module &m){

  py::class_<Reduction, IndexExpr>(m, "Reduction")
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

template<typename other_t, typename PyClass>
static void addIndexExprBinaryOps(PyClass &class_instance){

  class_instance
          .def("__add__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
            return self + IndexExpr(other);
          }, py::is_operator())

          .def("__radd__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return IndexExpr(other) + self;
          }, py::is_operator())

          .def("__sub__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return self - IndexExpr(other);
          }, py::is_operator())

          .def("__rsub__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return IndexExpr(other) - self;
          }, py::is_operator())

          .def("__mul__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return self * IndexExpr(other);
          }, py::is_operator())

          .def("__rmul__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return IndexExpr(other) * self;
          }, py::is_operator())

          .def("__div__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return new DivNode(self, IndexExpr(other), Datatype::Float64);
          }, py::is_operator())

          .def("__rdiv__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return new DivNode(IndexExpr(other), self, Datatype::Float64);
          }, py::is_operator())

          .def("__truediv__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return new DivNode(self, IndexExpr(other), Datatype::Float64);
          }, py::is_operator())

          .def("__rtruediv__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return new DivNode(IndexExpr(other), self, Datatype::Float64);
          }, py::is_operator())

          .def("__floordiv__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return new DivNode(self, IndexExpr(other), Datatype::Int64);
          }, py::is_operator())

          .def("__rfloordiv__", [](const IndexExpr &self, const other_t other) -> IndexExpr{
              return new DivNode(IndexExpr(other), self, Datatype::Int64);
          }, py::is_operator());



}

static void defineIndexExpr(py::module &m){

  py::class_<IndexExprNode, PyIndexExprNode> pyNode(m, "IndexExprNode");
  pyNode
      .def(py::init<>())
      .def("accept", &IndexExprNode::accept);

  auto exprClass = py::class_<IndexExpr>(m, "IndexExpr", pyNode)
          .def("datatype", &IndexExpr::getDataType)

          .def("equals", (bool (*) (IndexExpr, IndexExpr)) &equals)

          .def("__neg__", [](const IndexExpr &a) -> IndexExpr {
              return -a;
          }, py::is_operator())

          .def("__repr__", [](const IndexExpr &a) -> std::string {
              std::ostringstream o;
              if(a.defined()){
                o << "IndexExpr(" << a << ")";
              }else{
                o << a;
              }

              return o.str();
          }, py::is_operator());

  addIndexExprBinaryOps<IndexExpr>(exprClass);
  addIndexExprBinaryOps<int>(exprClass);
  addIndexExprBinaryOps<double>(exprClass);

  defineBinaryIndexExpr<Add>(m, "Add");
  defineBinaryIndexExpr<Sub>(m, "Sub");
  defineBinaryIndexExpr<Mul>(m, "Mul");
  defineBinaryIndexExpr<Div>(m, "Div");
  defineUnaryExpr<Sqrt>(m, "Sqrt");
  defineUnaryExpr<Neg>(m, "Neg");
  defineReduction(m);
}

static void defineAccess(py::module &m){

  py::class_<Access, IndexExpr>(m, "Access")
          .def(py::init<>())
          .def(py::init<TensorVar, std::vector<IndexVar>>(), py::arg("tensorVar"), py::arg("indices") = py::list())
          .def("tensor_var", &Access::getTensorVar)
          .def("index_vars", &Access::getIndexVars);

}

void defineIndexNotation(py::module &m){
  defineIndexVar(m);
  defineIndexExpr(m);
  defineAccess(m);
}


}}

