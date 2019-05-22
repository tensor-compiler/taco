#include "pyTensor.h"
#include "taco/tensor.h"
#include "pybind11/operators.h"
#include "taco/type.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

// Add Python dictionary initializer with {tuple(coordinate) : data} pairs


namespace taco{
namespace pythonBindings{

static void checkBounds(const std::vector<int>& dims, const std::vector<int>& indices){

  // Check for potential scalar access. Don't throw error if syntax valid
  if(dims.empty() && (indices.empty() || (indices[0] == 0 && indices.size() == 1))){
    return;
  }

  if(dims.size() != indices.size()){
    std::ostringstream o;
    o << "Incorrect number of dimensions when indexing. Tensor is order " << dims.size() << " but got index of "
                                                                                            "size " << indices.size();
    o << ". To index multiple dimensions only \"fancy\" notation is supported. For example to access the first "
         "element of a matrix, use A[0, 0] instead of A[0][0].";
    throw py::value_error(o.str());
  }

  for(size_t i = 0; i < dims.size(); ++i){
    if(indices[i] >= dims[i]){
      std::ostringstream o;
      o << "Index out of range for dimension " << i << ". Dimension shape is " << dims[i] << " but index value is "
           << indices[i];
      throw py::index_error(o.str());
    }
  }
}

template<typename T>
static Tensor<T> fromNumpy(py::array_t<T, py::array::f_style> array, bool copy) {
  py::buffer_info array_buffer = array.request();
  std::vector<ssize_t> buf_shape = array_buffer.shape;
  std::vector<int> shape(buf_shape.begin(), buf_shape.end());
  const ssize_t size = array.size();
  const ssize_t dims = array_buffer.ndim;

  // Creat col-major dense tensor
  std::vector<int> ordering;
  for(int i = dims-1; i >= 0; --i){
    ordering.push_back(i);
  }

  Format fmt(std::vector<ModeFormatPack>(dims, dense), ordering);
  Tensor<T> tensor(shape, fmt);

  TensorStorage& storage = tensor.getStorage();
  void *buf_data = array_buffer.ptr;
  Array::Policy policy = Array::Policy::UserOwns;
  if(copy){
    buf_data = new T[size];
    memcpy(buf_data, array_buffer.ptr, size*array_buffer.itemsize);
    policy = Array::Policy::Delete;
  }

  storage.setValues(makeArray(static_cast<T*>(buf_data), size, policy));
  tensor.setStorage(storage);
  return tensor;
}


template<typename T>
static Tensor<T> fromNumpy(py::array_t<T, py::array::c_style | py::array::forcecast>  array, bool copy) {

  py::buffer_info array_buffer = array.request();
  std::vector<ssize_t> buf_shape = array_buffer.shape;
  std::vector<int> shape(buf_shape.begin(), buf_shape.end());
  const ssize_t size = array.size();

  // Creat row-major dense tensor
  Tensor<T> tensor(shape, dense);
  TensorStorage& storage = tensor.getStorage();
  void *buf_data = array_buffer.ptr;
  Array::Policy policy = Array::Policy::UserOwns;
  if(copy){
    buf_data = new T[size];
    memcpy(buf_data, array_buffer.ptr, size*array_buffer.itemsize);
    policy = Array::Policy::Delete;
  }

  storage.setValues(makeArray(static_cast<T*>(buf_data), size, policy));
  tensor.setStorage(storage);
  return tensor;
}

template<typename IdxType, typename T>
static Tensor<T> fromSpMatrix(py::array_t<IdxType> ind_ptr, py::array_t<IdxType> inds, py::array_t<T> data,
                               const std::vector<int> &dims, bool copy, bool CSR){

  py::buffer_info ind_ptr_buf = ind_ptr.request();
  py::buffer_info inds_buf = inds.request();
  py::buffer_info data_buf = data.request();

  if(ind_ptr_buf.ndim != 1 || inds_buf.ndim != 1 || data_buf.ndim != 1) {
    throw py::value_error("Data arrays must be 1D.");
  }

  IdxType *mat_ptr  = static_cast<IdxType *>(ind_ptr_buf.ptr);
  IdxType *mat_ind  = static_cast<IdxType *>(inds_buf.ptr);
  T *mat_data = static_cast<T *>(data_buf.ptr);
  Array::Policy policy = Array::Policy::UserOwns;

  if(copy){
    mat_ptr = new IdxType[ind_ptr_buf.size];
    mat_ind = new IdxType[inds_buf.size];
    mat_data = new T[data_buf.size];
    memcpy(mat_ptr, ind_ptr_buf.ptr, ind_ptr_buf.size*ind_ptr_buf.itemsize);
    memcpy(mat_ind, inds_buf.ptr, inds_buf.size * inds_buf.itemsize);
    memcpy(mat_data, data_buf.ptr, data_buf.size * data_buf.itemsize);
    policy = Array::Policy::Delete;
  }

  // Create CSR Matrix
  Tensor<T> tensor;
  if(CSR){
    tensor = makeCSR(util::uniqueName("csr"), dims, mat_ptr, mat_ind, mat_data, policy);
  } else{
    tensor = makeCSC(util::uniqueName("csc"), dims, mat_ptr, mat_ind, mat_data, policy);
  }

  return tensor;
}

template<typename CType, typename idxVar>
static inline Access accessGetter(Tensor<CType>& tensor, idxVar& var) {
  return tensor(var);
}

template<typename CType>
static inline CType elementGetter(Tensor<CType>& tensor, std::vector<int> coords) {
  checkBounds(tensor.getDimensions(), coords);
  if(tensor.getOrder() == 0) {
    return tensor.at({});
  }
  return tensor.at(coords);
}

template<typename CType, typename pyType>
static inline void elementSetter(Tensor<CType> &tensor, std::vector<int> coords, pyType value) {
  checkBounds(tensor.getDimensions(), coords);
  if(tensor.getOrder() == 0) {
    tensor = static_cast<CType>(value);
  }
  tensor.insert(coords, static_cast<CType>(value));
}

template<typename CType, typename pyType>
static inline void singleElementSetter(Tensor<CType> &tensor, int coord, pyType value) {
  elementSetter<CType, pyType>(tensor, {coord}, value);
}

template<typename CType, typename VarType, typename ExprType>
static inline void exprSetter(Tensor<CType> &tensor, VarType idx, ExprType expr) {
  tensor(idx) = expr;
}

template<typename CType>
static void declareTensor(py::module &m, const std::string typestr) {
  using typedTensor = Tensor<CType>;

  m.def("fromNpF", (typedTensor (*)(py::array_t<CType, py::array::f_style> array, bool copy))
                             &fromNumpy<CType>, py::arg("array").noconvert(), py::arg("copy"));
  m.def("fromNpC", (typedTensor (*)(py::array_t<CType, py::array::c_style |
                                              py::array::forcecast> array, bool copy)) &fromNumpy<CType>);

  m.def("fromSpMatrix", &fromSpMatrix<int, CType>);

  std::string pyClassName = std::string("Tensor") + typestr;
  py::class_<typedTensor, TensorBase>(m, pyClassName.c_str(), py::buffer_protocol())

          .def(py::init<>())

          .def(py::init<TensorBase>())

          .def(py::init<std::string>(), py::arg("name"))

          .def(py::init<CType>(), py::arg("value"))

          .def(py::init<std::string, std::vector<int>, ModeFormat>(), py::arg("name"), py::arg("shape"),
               py::arg("format") = ModeFormat::compressed)

          .def(py::init<std::string, std::vector<int>, Format>(), py::arg("name"), py::arg("shape"),
               py::arg("format"))

          .def_buffer([](typedTensor &t) -> py::buffer_info {

              if(!isDense(t.getFormat())){
                throw py::value_error("Cannot export a compressed tensor. Make sure all dimensions are dense "
                                      "using to_dense() before attempting this conversion.");
              }

              // Force computation of the tensor
              t.pack();
              if(t.needsCompute()){
                t.evaluate();
              }

              void *ptr = t.getStorage().getValues().getData();

              std::vector<ssize_t> shape (t.getDimensions().begin(), t.getDimensions().end());
              std::vector<ssize_t> row_major_strides;

              for(size_t i = 0; i < shape.size(); ++i) {
                ssize_t currentStride = sizeof(CType);
                for(size_t j = i + 1; j < shape.size(); ++j){
                  currentStride *= shape[j];
                }
                row_major_strides.push_back(currentStride);
              }

              std::vector<ssize_t> strides;
              for(const int &permutation : t.getFormat().getModeOrdering()){
                strides.push_back(row_major_strides[permutation]);
              }

              return py::buffer_info(
                      ptr,                                         /* Pointer to buffer */
                      sizeof(CType),                               /* Size of one scalar */
                      py::format_descriptor<CType>::format(),      /* Python struct-style format descriptor */
                      t.getOrder(),                                /* Number of dimensions */
                      shape,                                       /* Buffer dimensions */
                      strides                                      /* Strides (in bytes) for each index */
              );
          })

          .def("set_name", &TensorBase::setName)

          .def("get_name", &TensorBase::getName)

          .def("order", &TensorBase::getOrder)

          .def("get_shape", &TensorBase::getDimension, py::arg("axis"))

          .def("dtype", &TensorBase::getComponentType)

          .def("get_dimensions", &TensorBase::getDimensions)

          .def("format", &TensorBase::getFormat)

          .def("pack", &typedTensor::pack)

          .def("compile", &typedTensor::compile)

          .def("assemble", &typedTensor::assemble)

          .def("evaluate", &typedTensor::evaluate)

          .def("compute", &typedTensor::compute)

          .def("transpose", [](typedTensor &self, std::vector<int> dims, Format format, std::string name) -> void{
              self.transpose(name, dims, format);
          }, py::is_operator())

          .def("__getitem__", [](typedTensor& self, const int &index) -> CType {
               return elementGetter<CType>(self, {index});
            }, py::is_operator())

          .def("__getitem__", [](typedTensor& self, const std::vector<int> &indices) -> CType {
               return elementGetter<CType>(self, indices);
            }, py::is_operator())

          .def("__getitem__", [](typedTensor& self, nullptr_t ptr) -> Access{
            if(self.getOrder() != 0) {
              throw py::index_error("Can only index scalar tensors with None.");
            }
            return self();
          }, py::is_operator())

          .def("__getitem__", &accessGetter<CType, IndexVar&>, py::is_operator())

          .def("__getitem__", &accessGetter<CType, std::vector<IndexVar>&>, py::is_operator())

          .def("__setitem__", &singleElementSetter<CType, int64_t>, py::is_operator())

          .def("__setitem__", &singleElementSetter<CType, double>, py::is_operator())

          .def("__setitem__", &elementSetter<CType, int64_t>, py::is_operator())

          .def("__setitem__", &elementSetter<CType, double>, py::is_operator())

          .def("__setitem__", [](typedTensor& self, nullptr_t ptr, int64_t value) -> void {
              if(self.getOrder() != 0) {
                throw py::index_error("Can only index scalar tensors with None.");
              }
              self = static_cast<CType>(value);
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, nullptr_t ptr, double value) -> void {
              if(self.getOrder() != 0) {
                throw py::index_error("Can only index scalar tensors with None.");
              }
              self = static_cast<CType>(value);
          }, py::is_operator())

          //  Set scalars to expression using none
          .def("__setitem__", [](typedTensor& self, nullptr_t ptr, const IndexExpr expr) -> void {
              self = expr;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, nullptr_t ptr, const Access access) -> void {
              self = access;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, nullptr_t ptr, const TensorVar tensorVar) -> void {
              self = tensorVar;
          }, py::is_operator())

          // Set expressions with varying types
          .def("__setitem__", &exprSetter<CType, IndexVar, IndexExpr>, py::is_operator())

          .def("__setitem__", &exprSetter<CType, IndexVar, Access>, py::is_operator())

          .def("__setitem__", &exprSetter<CType, IndexVar, TensorVar>, py::is_operator())

          .def("__setitem__", &exprSetter<CType, std::vector<IndexVar>, IndexExpr>, py::is_operator())

          .def("__setitem__", &exprSetter<CType, std::vector<IndexVar>, Access>, py::is_operator())

          .def("__setitem__", &exprSetter<CType, std::vector<IndexVar>, TensorVar>, py::is_operator())

          // This is a hack that exploits pybind11's resolution order. If we get here all other methods to resolve the
          // function failed and we throw an error. There may be better was to handle this in pybind.
          .def("__getitem__", [](typedTensor& self, const py::object &indices) -> void {
              throw py::index_error("Indices must be an iterable of integers or IndexVars");
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const py::object &indices, py::object value) -> void {
              throw py::index_error("Indices must be an iterable of integers or IndexVars");
          }, py::is_operator())

          .def("__repr__",   [](typedTensor& self) -> std::string{
              std::ostringstream o;
              o << self;
              return o.str();
          }, py::is_operator());

}

void defineTensor(py::module &m) {

  py::class_<TensorBase>(m, "TensorBase")
          .def("dtype", &TensorBase::getComponentType);

  declareTensor<int8_t>(m, "Int8");
  declareTensor<int16_t>(m, "Int16");
  declareTensor<int32_t>(m, "Int32");
  declareTensor<int64_t>(m, "Int64");
  declareTensor<uint8_t>(m, "UInt8");
  declareTensor<uint16_t>(m, "UInt16");
  declareTensor<uint32_t>(m, "UInt32");
  declareTensor<uint64_t>(m, "UInt64");
  declareTensor<float>(m, "Float");
  declareTensor<double>(m, "Double");
}

}}


