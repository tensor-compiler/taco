#include "taco/tensor.h"

#include <set>
#include <cstring>
#include <fstream>
#include <sstream>
#include <limits.h>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/format.h"
#include "taco/ir/ir.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"
#include "taco/storage/pack.h"
#include "taco/ir/ir.h"
#include "taco/lower/lower.h"
#include "taco/lower/schedule.h"
#include "lower/iteration_graph.h"
#include "backends/module.h"
#include "taco/taco_tensor_t.h"
#include "taco/io/tns_file_format.h"
#include "taco/io/mtx_file_format.h"
#include "taco/io/rb_file_format.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"
#include "taco/util/name_generator.h"
#include "error/error_messages.h"
#include "error/error_checks.h"

using namespace std;
using namespace taco::ir;
using namespace taco::storage;
using namespace taco::expr_nodes;

namespace taco {

static const size_t DEFAULT_ALLOC_SIZE = (1 << 20);

struct TensorBase::Content {
  string                name;
  vector<int>           dimensions;
  Type                  ctype;

  storage::Storage      storage;

  vector<IndexVar>      indexVars;
  IndexExpr             expr;
  bool                  accumulate;  // Accumulate expr into result (+=)

  lower::Schedule       schedule;

  vector<void*>         arguments;

  size_t                allocSize;
  size_t                valuesSize;

  Stmt                  assembleFunc;
  Stmt                  computeFunc;
  bool                  assembleWhileCompute;
  shared_ptr<Module>    module;
};

TensorBase::TensorBase() : TensorBase(Float(64)) {
}

TensorBase::TensorBase(Type ctype)
    : TensorBase(util::uniqueName('A'), ctype) {
}

TensorBase::TensorBase(std::string name, Type ctype)
    : TensorBase(name, ctype, {}, Format())  {
}

TensorBase::TensorBase(double val) : TensorBase(type<double>()) {
  this->insert({}, val);
  pack();
}

TensorBase::TensorBase(Type ctype, vector<int> dimensions, Format format)
    : TensorBase(util::uniqueName('A'), ctype, dimensions, format) {
}

TensorBase::TensorBase(string name, Type ctype, vector<int> dimensions,
                       Format format) : content(new Content) {
  taco_uassert(format.getOrder() == dimensions.size() ||
               format.getOrder() == 1) <<
      "The number of format mode types (" << format.getOrder() << ") " <<
      "must match the tensor order (" << dimensions.size() << "), " <<
      "or there must be a single mode type.";

  if (dimensions.size() == 0) {
    format = Format();
  }
  else if (dimensions.size() > 1 && format.getOrder() == 1) {
    ModeType levelType = format.getModeTypes()[0];
    vector<ModeType> levelTypes;
    for (size_t i = 0; i < dimensions.size(); i++) {
      levelTypes.push_back(levelType);
    }
    format = Format(levelTypes);
  }

  content->name = name;
  content->dimensions = dimensions;
  content->storage = Storage(format);
  content->ctype = ctype;
  this->setAllocSize(DEFAULT_ALLOC_SIZE);

  // Initialize dense storage modes
  // TODO: Get rid of this and make code use dimensions instead of dense indices
  vector<ModeIndex> modeIndices(format.getOrder());
  for (size_t i = 0; i < format.getOrder(); ++i) {
    if (format.getModeTypes()[i] == ModeType::Dense) {
      const size_t idx = format.getModeOrdering()[i];
      modeIndices[i] = ModeIndex({makeArray({dimensions[idx]})});
    }
  }
  content->storage.setIndex(Index(format, modeIndices));

  content->assembleWhileCompute = false;
  content->module = make_shared<Module>();

  this->coordinateBuffer = shared_ptr<vector<char>>(new vector<char>);
  this->coordinateBufferUsed = 0;
  this->coordinateSize = getOrder()*sizeof(int) + ctype.getNumBytes();
}

void TensorBase::setName(std::string name) const {
  content->name = name;
}

string TensorBase::getName() const {
  return content->name;
}

size_t TensorBase::getOrder() const {
  return content->dimensions.size();
}

int TensorBase::getDimension(size_t mode) const {
  taco_uassert(mode < getOrder()) << "Invalid mode";
  return content->dimensions[mode];
}

const vector<int>& TensorBase::getDimensions() const {
  return content->dimensions;
}

const Format& TensorBase::getFormat() const {
  return content->storage.getFormat();
}

void TensorBase::reserve(size_t numCoordinates) {
  size_t newSize = this->coordinateBuffer->size() +
                   numCoordinates*this->coordinateSize;
  this->coordinateBuffer->resize(newSize);
}

void TensorBase::insert(const initializer_list<int>& coordinate, double value) {
  taco_uassert(coordinate.size() == getOrder()) <<
      "Wrong number of indices";
  taco_uassert(getComponentType() == Float(64)) <<
      "Cannot insert a value of type '" << Float(64) << "' " <<
      "into a tensor with component type " << getComponentType();
  if ((coordinateBuffer->size() - coordinateBufferUsed) < coordinateSize) {
    coordinateBuffer->resize(coordinateBuffer->size() + coordinateSize);
  }
  int* coordLoc = (int*)&coordinateBuffer->data()[coordinateBufferUsed];
  for (int idx : coordinate) {
    *coordLoc = idx;
    coordLoc++;
  }
  *((double*)coordLoc) = value;
  coordinateBufferUsed += coordinateSize;
}

void TensorBase::insert(const std::vector<int>& coordinate, double value) {
  taco_uassert(coordinate.size() == getOrder()) <<
      "Wrong number of indices";
  taco_uassert(getComponentType() == Float(64)) <<
      "Cannot insert a value of type '" << Float(64) << "' " <<
      "into a tensor with component type " << getComponentType();
  if ((coordinateBuffer->size() - coordinateBufferUsed) < coordinateSize) {
    coordinateBuffer->resize(coordinateBuffer->size() + coordinateSize);
  }
  int* coordLoc = (int*)&coordinateBuffer->data()[coordinateBufferUsed];
  for (int idx : coordinate) {
    *coordLoc = idx;
    coordLoc++;
  }
  *((double*)coordLoc) = value;
  coordinateBufferUsed += coordinateSize;
}

const Type& TensorBase::getComponentType() const {
  return content->ctype;
}

const vector<IndexVar>& TensorBase::getIndexVars() const {
  return content->indexVars;
}

const IndexExpr& TensorBase::getExpr() const {
  return content->expr;
}

const storage::Storage& TensorBase::getStorage() const {
  return content->storage;
}

storage::Storage& TensorBase::getStorage() {
  return content->storage;
}

void TensorBase::setAllocSize(size_t allocSize) {
  taco_uassert(allocSize >= 2 && (allocSize & (allocSize - 1)) == 0) <<
      "The index allocation size must be a power of two and at least two";
  content->allocSize = allocSize;
}

size_t TensorBase::getAllocSize() const {
  return content->allocSize;
}

static size_t numIntegersToCompare = 0;
static int lexicographicalCmp(const void* a, const void* b) {
  for (size_t i = 0; i < numIntegersToCompare; i++) {
    int diff = ((int*)a)[i] - ((int*)b)[i];
    if (diff != 0) {
      return diff;
    }
  }
  return 0;
}

/// Pack coordinates into a data structure given by the tensor format.
void TensorBase::pack() {
  taco_tassert(getComponentType().getKind() == Type::Float &&
               getComponentType().getNumBits() == 64)
      << "make the packing machinery work with other primitive types later. "
      << "Right now we're specializing to doubles so that we can use a "
      << "resizable vector, but eventually we should use a two pass pack "
      << "algorithm that figures out sizes first, and then packs the data";

  const size_t order = getOrder();


  // Pack scalars
  if (order == 0) {
    char* coordLoc = this->coordinateBuffer->data();
    double scalarValue = *(double*)&coordLoc[this->coordinateSize -
                                             getComponentType().getNumBytes()];
    content->storage.setValues(makeArray({scalarValue}));
    this->coordinateBuffer->clear();
    return;
  }


  /// Permute the coordinates according to the storage mode ordering.
  /// This is a workaround since the current pack code only packs tensors in the
  /// ordering of the modes.
  const std::vector<int>& dimensions = getDimensions();
  taco_iassert(getFormat().getOrder() == order);
  std::vector<size_t> permutation = getFormat().getModeOrdering();
  std::vector<int> permutedDimensions(order);
  for (size_t i = 0; i < order; ++i) {
    permutedDimensions[i] = dimensions[permutation[i]];
  }

  taco_iassert((this->coordinateBufferUsed % this->coordinateSize) == 0);
  size_t numCoordinates = this->coordinateBufferUsed / this->coordinateSize;
  const size_t coordSize = this->coordinateSize;

  char* coordinatesPtr = coordinateBuffer->data();
  vector<int> permuteBuffer(order);
  for (size_t i=0; i < numCoordinates; ++i) {
    int* coordinate = (int*)coordinatesPtr;
    for (size_t j = 0; j < order; j++) {
      permuteBuffer[j] = coordinate[permutation[j]];
    }
    for (size_t j = 0; j < order; j++) {
      coordinate[j] = permuteBuffer[j];
    }
    coordinatesPtr += this->coordinateSize;
  }
  coordinatesPtr = coordinateBuffer->data();


  // The pack code expects the coordinates to be sorted
  numIntegersToCompare = order;
  qsort(coordinatesPtr, numCoordinates, coordSize, lexicographicalCmp);


  // Move coords into separate arrays and remove duplicates
  std::vector<std::vector<int>> coordinates(order);
  for (size_t i=0; i < order; ++i) {
    coordinates[i] = std::vector<int>(numCoordinates);
  }
  std::vector<double> values(numCoordinates);
  // Copy first coordinate-value pair
  int* lastCoord = (int*)malloc(order * sizeof(int));
  if (numCoordinates >= 1) {
    int* coordComponent = (int*)coordinatesPtr;
    for (size_t d=0; d < order; ++d) {
      coordinates[d][0] = *coordComponent;
      lastCoord[d] = *coordComponent;
      coordComponent++;
    }
    values[0] = *((double*)coordComponent);
  }
  // Copy remaining coordinate-value pairs, removing duplicates
  int j = 1;
  int* coord = (int*)malloc(order * sizeof(int));
  for (size_t i=1; i < numCoordinates; ++i) {
    int* coordLoc = (int*)&coordinatesPtr[i*coordSize];
    for (size_t d=0; d < order; ++d) {
      coord[d] = *coordLoc;;
      coordLoc++;
    }
    double value = *((double*)coordLoc);
    if (memcmp(coord, lastCoord, order*sizeof(int)) != 0) {
      for (size_t d = 0; d < order; d++) {
        coordinates[d][j] = coord[d];
      }
      values[j] = value;
      j++;
    }
    else {
      values[j-1] += value;
    }
  }
  free(coord);
  free(lastCoord);
  if (numCoordinates > 0) {
    for (size_t i=0; i < order; ++i) {
      coordinates[i].resize(j);
    }
    values.resize(j);
  }
  taco_iassert(coordinates.size() > 0);
  this->coordinateBuffer->clear();
  this->coordinateBufferUsed = 0;

  // Pack indices and values
  content->storage = storage::pack(permutedDimensions, getFormat(),
                                   coordinates, values);

//  std::cout << storage::packCode(getFormat()) << std::endl;
}

void TensorBase::zero() {
  getStorage().getValues().zero();
}

const Access TensorBase::operator()(const std::vector<IndexVar>& indices) const {
  taco_uassert(indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return Access(*this, indices);
}

Access TensorBase::operator()(const std::vector<IndexVar>& indices) {
  taco_uassert(indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return Access(*this, indices);
}

void TensorBase::compile(bool assembleWhileCompute) {
  taco_uassert(getExpr().defined()) << error::compile_without_expr;

  std::set<lower::Property> assembleProperties, computeProperties;
  assembleProperties.insert(lower::Assemble);
  computeProperties.insert(lower::Compute);
  if (content->accumulate) {
    computeProperties.insert(lower::Accumulate);
  }
  if (assembleWhileCompute) {
    computeProperties.insert(lower::Assemble);
  }

  content->assembleWhileCompute = assembleWhileCompute;
  content->assembleFunc = lower::lower(*this, "assemble", content->schedule,
                                       assembleProperties);
  content->computeFunc  = lower::lower(*this, "compute", content->schedule,
                                       computeProperties);
  content->module->addFunction(content->assembleFunc);
  content->module->addFunction(content->computeFunc);
  content->module->compile();
}

static taco_tensor_t* packTensorData(const TensorBase& tensor) {
  taco_tensor_t* tensorData = (taco_tensor_t*)malloc(sizeof(taco_tensor_t));
  size_t order = tensor.getOrder();
  Storage storage = tensor.getStorage();
  Format format = storage.getFormat();

  taco_iassert(order <= INT_MAX);
  tensorData->order         = static_cast<int>(order);
  tensorData->dimensions    = (int32_t*)malloc(order * sizeof(int32_t));
  tensorData->mode_ordering = (int32_t*)malloc(order * sizeof(int32_t));
  tensorData->mode_types    = (taco_mode_t*)malloc(order * sizeof(taco_mode_t));
  tensorData->indices       = (uint8_t***)malloc(order * sizeof(uint8_t***));

  auto index = storage.getIndex();
  for (size_t i = 0; i < tensor.getOrder(); i++) {
    auto modeType  = format.getModeTypes()[i];
    auto modeIndex = index.getModeIndex(i);

    tensorData->dimensions[i] = tensor.getDimension(i);

    size_t m = format.getModeOrdering()[i];
    taco_iassert(m <= INT_MAX);
    tensorData->mode_ordering[i] = static_cast<int>(m);

    switch (modeType) {
      case ModeType::Dense: {
        tensorData->mode_types[i] = taco_mode_dense;
        tensorData->indices[i]    = (uint8_t**)malloc(1 * sizeof(uint8_t**));

        const Array& size = modeIndex.getIndexArray(0);
        tensorData->indices[i][0] = (uint8_t*)size.getData();
        break;
      }
      case ModeType::Sparse: {
        tensorData->mode_types[i]  = taco_mode_sparse;
        tensorData->indices[i]    = (uint8_t**)malloc(2 * sizeof(uint8_t**));

        // When packing results for assemblies they won't have sparse indices
        if (modeIndex.numIndexArrays() == 0) {
          continue;
        }

        const Array& pos = modeIndex.getIndexArray(0);
        const Array& idx = modeIndex.getIndexArray(1);
        tensorData->indices[i][0] = (uint8_t*)pos.getData();
        tensorData->indices[i][1] = (uint8_t*)idx.getData();
      }
        break;
      case ModeType::Fixed:
        taco_not_supported_yet;
        break;
    }
  }

  taco_iassert(tensor.getComponentType().getNumBits() <= INT_MAX);
  tensorData->csize = static_cast<int>(tensor.getComponentType().getNumBits());
  tensorData->vals  = (uint8_t*)storage.getValues().getData();

  return tensorData;
}

taco_tensor_t* TensorBase::getTacoTensorT() {
  return packTensorData(*this);
}

static size_t unpackTensorData(const taco_tensor_t& tensorData,
                               const TensorBase& tensor) {
  auto storage = tensor.getStorage();
  auto format = storage.getFormat();

  vector<ModeIndex> modeIndices;
  size_t numVals = 1;
  for (size_t i = 0; i < tensor.getOrder(); i++) {
    ModeType modeType = format.getModeTypes()[i];
    switch (modeType) {
      case ModeType::Dense: {
        Array size = makeArray({*(int*)tensorData.indices[i][0]});
        modeIndices.push_back(ModeIndex({size}));
        numVals *= ((int*)tensorData.indices[i][0])[0];
        break;
      }
      case ModeType::Sparse: {
        auto size = ((int*)tensorData.indices[i][0])[numVals];
        Array pos = Array(type<int>(), tensorData.indices[i][0], numVals+1);
        Array idx = Array(type<int>(), tensorData.indices[i][1], size);
        modeIndices.push_back(ModeIndex({pos, idx}));
        numVals = size;
        break;
      }
      case ModeType::Fixed:
        taco_not_supported_yet;
        break;
    }
  }
  storage.setIndex(Index(format, modeIndices));
  storage.setValues(Array(type<double>(), tensorData.vals, numVals));
  return numVals;
}

static inline
vector<void*> packArguments(const TensorBase& tensor) {
  vector<void*> arguments;

  // Pack the result tensor
  arguments.push_back(packTensorData(tensor));

  // Pack operand tensors
  vector<TensorBase> operands = expr_nodes::getOperands(tensor.getExpr());
  for (auto& operand : operands) {
    arguments.push_back(packTensorData(operand));
  }

  return arguments;
}

void TensorBase::assemble() {
  taco_uassert(this->content->assembleFunc.defined())
      << error::assemble_without_compile;

  this->content->arguments = packArguments(*this);
  content->module->callFuncPacked("assemble", content->arguments.data());

  if (!content->assembleWhileCompute) {
    taco_tensor_t* tensorData = ((taco_tensor_t*)content->arguments[0]);
    content->valuesSize = unpackTensorData(*tensorData, *this);
  }
}

void TensorBase::compute() {
  taco_uassert(this->content->computeFunc.defined())
      << error::compute_without_compile;

  this->content->arguments = packArguments(*this);
  this->content->module->callFuncPacked("compute", content->arguments.data());

  if (content->assembleWhileCompute) {
    taco_tensor_t* tensorData = ((taco_tensor_t*)content->arguments[0]);
    content->valuesSize = unpackTensorData(*tensorData, *this);
  }
}

void TensorBase::evaluate() {
  this->compile();
  if (!content->accumulate) {
    this->assemble();
  }
  this->compute();
}

void TensorBase::setExpr(const vector<IndexVar>& indexVars, IndexExpr expr,
                         bool accumulate) {
  taco_uassert(error::dimensionsTypecheck(indexVars, expr, getDimensions()))
      << error::expr_dimension_mismatch << " "
      << error::dimensionTypecheckErrors(indexVars, expr, getDimensions());

  // The following are index expressions the implementation doesn't currently
  // support, but that are planned for the future.
  taco_uassert(!error::containsTranspose(this->getFormat(), indexVars, expr))
      << error::expr_transposition;
  taco_uassert(!error::containsDistribution(indexVars, expr))
      << error::expr_distribution;

  content->indexVars  = indexVars;
  content->expr       = expr;
  content->accumulate = accumulate;
}

void TensorBase::printComputeIR(ostream& os, bool color, bool simplify) const {
  IRPrinter printer(os, color, simplify);
  printer.print(content->computeFunc.as<Function>()->body);
}

void TensorBase::printAssembleIR(ostream& os, bool color, bool simplify) const {
  IRPrinter printer(os, color, simplify);
  printer.print(content->assembleFunc.as<Function>()->body);
}

string TensorBase::getSource() const {
  return content->module->getSource();
}

void TensorBase::compileSource(std::string source) {
  taco_iassert(getExpr().defined()) << "No expression defined for tensor";

  set<lower::Property> assembleProperties, computeProperties;
  assembleProperties.insert(lower::Assemble);
  computeProperties.insert(lower::Compute);
  if (content->accumulate) {
    computeProperties.insert(lower::Accumulate);
  }

  content->assembleFunc = lower::lower(*this, "assemble", content->schedule,
                                       assembleProperties);
  content->computeFunc  = lower::lower(*this, "compute", content->schedule,
                                       computeProperties);

  stringstream ss;
  CodeGen_C::generateShim(content->assembleFunc, ss);
  ss << endl;
  CodeGen_C::generateShim(content->computeFunc, ss);
  content->module->setSource(source + "\n" + ss.str());
  content->module->compile();
}

bool equals(const TensorBase& a, const TensorBase& b) {
  // Component type must be the same
  if (a.getComponentType() != b.getComponentType()) {
    return false;
  }

  // Orders must be the same
  if (a.getOrder() != b.getOrder()) {
    return false;
  }

  // Dimensions must be the same
  for (size_t mode = 0; mode < a.getOrder(); mode++) {
    if (a.getDimension(mode) != b.getDimension(mode)) {
      return false;
    }
  }

  // Values must be the same
  auto at = iterate<double>(a);
  auto bt = iterate<double>(b);
  auto ait = at.begin();
  auto bit = bt.begin();

  for (; ait != at.end() && bit != bt.end(); ++ait, ++bit) {
    if (ait->first != bit->first) {
      return false;
    }
    if (abs((ait->second - bit->second)/ait->second) > 10e-6) {
      return false;
    }
  }

  return (ait == at.end() && bit == bt.end());
}

bool operator==(const TensorBase& a, const TensorBase& b) {
  return a.content == b.content;
}

bool operator!=(const TensorBase& a, const TensorBase& b) {
  return a.content != b.content;
}

bool operator<(const TensorBase& a, const TensorBase& b) {
  return a.content < b.content;
}

bool operator>(const TensorBase& a, const TensorBase& b) {
  return a.content > b.content;
}

bool operator<=(const TensorBase& a, const TensorBase& b) {
  return a.content <= b.content;
}

bool operator>=(const TensorBase& a, const TensorBase& b) {
  return a.content >= b.content;
}

ostream& operator<<(ostream& os, const TensorBase& tensor) {
  vector<string> dimensionStrings;
  for (int dimension : tensor.getDimensions()) {
    dimensionStrings.push_back(to_string(dimension));
  }
  os << tensor.getName() << " (" << util::join(dimensionStrings, "x") << ") "
     << tensor.getFormat() << ":" << std::endl;

  // Print coordinates
  size_t numCoordinates = tensor.coordinateBufferUsed / tensor.coordinateSize;
  for (size_t i = 0; i < numCoordinates; i++) {
    int* ptr = (int*)&tensor.coordinateBuffer->data()[i*tensor.coordinateSize];
    os << "(" << util::join(ptr, ptr+tensor.getOrder()) << "): "
       << ((double*)(ptr+tensor.getOrder()))[0] << std::endl;
  }

  // Print packed data
  os << tensor.getStorage();

  return os;
}

static string getExtension(string filename) {
  return filename.substr(filename.find_last_of(".") + 1);
}

template <typename T>
TensorBase dispatchRead(T& file, FileType filetype, Format format, bool pack) {
  TensorBase tensor;
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      tensor = io::mtx::read(file, format, pack);
      break;
    case FileType::tns:
      tensor = io::tns::read(file, format, pack);
      break;
    case FileType::rb:
      tensor = io::rb::read(file, format, pack);
      break;
  }
  return tensor;
}

TensorBase read(std::string filename, Format format, bool pack) {
  string extension = getExtension(filename);

  TensorBase tensor;
  if (extension == "ttx") {
    tensor = dispatchRead(filename, FileType::ttx, format, pack);
  }
  else if (extension == "tns") {
    tensor = dispatchRead(filename, FileType::tns, format, pack);
  }
  else if (extension == "mtx") {
    tensor = dispatchRead(filename, FileType::mtx, format, pack);
  }
  else if (extension == "rb") {
    tensor = dispatchRead(filename, FileType::rb, format, pack);
  }
  else {
    taco_uerror << "File extension not recognized: " << filename << std::endl;
  }

  string name = filename.substr(filename.find_last_of("/") + 1);
  name = name.substr(0, name.find_first_of("."));
  std::replace(name.begin(), name.end(), '-', '_');
  tensor.setName(name);

  return tensor;
}

TensorBase read(string filename, FileType filetype, Format format, bool pack) {
  return dispatchRead(filename, filetype, format, pack);
}

TensorBase read(istream& stream, FileType filetype,  Format format, bool pack) {
  return dispatchRead(stream, filetype, format, pack);
}

template <typename T>
void dispatchWrite(T& file, const TensorBase& tensor, FileType filetype) {
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      io::mtx::write(file, tensor);
      break;
    case FileType::tns:
      io::tns::write(file, tensor);
      break;
    case FileType::rb:
      io::rb::write(file, tensor);
      break;
  }
}

void write(string filename, const TensorBase& tensor) {
  string extension = getExtension(filename);
  if (extension == "ttx") {
    dispatchWrite(filename, tensor, FileType::ttx);
  }
  else if (extension == "tns") {
    dispatchWrite(filename, tensor, FileType::tns);
  }
  else if (extension == "mtx") {
    taco_iassert(tensor.getOrder() == 2) <<
       "The .mtx format only supports matrices. Consider using the .ttx format "
       "instead";
    dispatchWrite(filename, tensor, FileType::mtx);
  }
  else if (extension == "rb") {
    dispatchWrite(filename, tensor, FileType::rb);
  }
  else {
    taco_uerror << "File extension not recognized: " << filename << std::endl;
  }
}

void write(string filename, FileType filetype, const TensorBase& tensor) {
  dispatchWrite(filename, tensor, filetype);
}

void write(ofstream& stream, FileType filetype, const TensorBase& tensor) {
  dispatchWrite(stream, tensor, filetype);
}

TensorBase makeCSR(const std::string& name, const std::vector<int>& dimensions,
                   int* rowptr, int* colidx, double* vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<double> tensor(name, dimensions, CSR);
  auto storage = tensor.getStorage();
  auto index = storage::makeCSRIndex(dimensions[0], rowptr, colidx);
  storage.setIndex(index);
  storage.setValues(storage::makeArray(vals, index.getSize(), Array::UserOwns));
  return tensor;
}

TensorBase makeCSR(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& rowptr,
                   const std::vector<int>& colidx,
                   const std::vector<double>& vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<double> tensor(name, dimensions, CSR);
  auto storage = tensor.getStorage();
  storage.setIndex(storage::makeCSRIndex(rowptr, colidx));
  storage.setValues(storage::makeArray(vals));
  return tensor;
}

void getCSRArrays(const TensorBase& tensor,
                  int** rowptr, int** colidx, double** vals) {
  taco_uassert(tensor.getFormat() == CSR) <<
      "The tensor " << tensor.getName() << " is not defined in the CSR format";
  auto storage = tensor.getStorage();
  auto index = storage.getIndex();

  auto rowptrArr = index.getModeIndex(1).getIndexArray(0);
  auto colidxArr = index.getModeIndex(1).getIndexArray(1);
  taco_uassert(rowptrArr.getType() == type<int>()) << error::type_mismatch;
  taco_uassert(colidxArr.getType() == type<int>()) << error::type_mismatch;
  *rowptr = static_cast<int*>(rowptrArr.getData());
  *colidx = static_cast<int*>(colidxArr.getData());
  *vals   = static_cast<double*>(storage.getValues().getData());
}

TensorBase makeCSC(const std::string& name, const std::vector<int>& dimensions,
                   int* colptr, int* rowidx, double* vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<double> tensor(name, dimensions, CSC);
  auto storage = tensor.getStorage();
  auto index = storage::makeCSCIndex(dimensions[1], colptr, rowidx);
  storage.setIndex(index);
  storage.setValues(storage::makeArray(vals, index.getSize(), Array::UserOwns));
  return tensor;
}

TensorBase makeCSC(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& colptr,
                   const std::vector<int>& rowidx,
                   const std::vector<double>& vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<double> tensor(name, dimensions, CSC);
  auto storage = tensor.getStorage();
  storage.setIndex(storage::makeCSCIndex(colptr, rowidx));
  storage.setValues(storage::makeArray(vals));
  return tensor;
}

void getCSCArrays(const TensorBase& tensor,
                  int** colptr, int** rowidx, double** vals) {
  taco_uassert(tensor.getFormat() == CSC) <<
      "The tensor " << tensor.getName() << " is not defined in the CSC format";
  auto storage = tensor.getStorage();
  auto index = storage.getIndex();

  auto colptrArr = index.getModeIndex(1).getIndexArray(0);
  auto rowidxArr = index.getModeIndex(1).getIndexArray(1);
  taco_uassert(colptrArr.getType() == type<int>()) << error::type_mismatch;
  taco_uassert(rowidxArr.getType() == type<int>()) << error::type_mismatch;
  *colptr = static_cast<int*>(colptrArr.getData());
  *rowidx = static_cast<int*>(rowidxArr.getData());
  *vals   = static_cast<double*>(storage.getValues().getData());
}

void packOperands(const TensorBase& tensor) {
  for (TensorBase operand : expr_nodes::getOperands(tensor.getExpr())) {
    operand.pack();
  }
}

}
