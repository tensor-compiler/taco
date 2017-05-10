#include "taco/tensor.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <limits.h>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/format.h"
#include "ir/ir.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "taco/storage/storage.h"
#include "taco/storage/pack.h"
#include "ir/ir.h"
#include "lower/lower.h"
#include "lower/iteration_schedule.h"
#include "backends/module.h"
#include "taco/io/dns_file_format.h"
#include "taco/io/tns_file_format.h"
#include "taco/io/mtx_file_format.h"
#include "taco/io/rb_file_format.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"
#include "taco/util/name_generator.h"

using namespace std;
using namespace taco::ir;
using namespace taco::storage;
using namespace taco::expr_nodes;

namespace taco {

// class ComponentType
size_t ComponentType::bytes() const {
  switch (this->kind) {
    case Bool:
      return sizeof(bool);
    case Int:
      return sizeof(int);
    case Float:
      return sizeof(float);
    case Double:
      return sizeof(double);
    case Unknown:
      break;
  }
  return UINT_MAX;
}

ComponentType::Kind ComponentType::getKind() const {
  return kind;
}

bool operator==(const ComponentType& a, const ComponentType& b) {
  return a.getKind() == b.getKind();
}

bool operator!=(const ComponentType& a, const ComponentType& b) {
  return a.getKind() != b.getKind();
}

std::ostream& operator<<(std::ostream& os, const ComponentType& type) {
  switch (type.getKind()) {
    case ComponentType::Bool:
      os << "bool";
      break;
    case ComponentType::Int:
      os << "int";
      break;
    case ComponentType::Float:
      os << "float";
      break;
    case ComponentType::Double:
      os << "double";
      break;
    case ComponentType::Unknown:
      break;
  }
  return os;
}


static const size_t DEFAULT_ALLOC_SIZE = (1 << 20);

struct TensorBase::Content {
  string                   name;
  vector<int>              dimensions;
  ComponentType            ctype;

  storage::Storage         storage;

  vector<taco::Var>        indexVars;
  taco::Expr               expr;
  vector<void*>            arguments;

  size_t                   allocSize;
  size_t                   valuesSize;

  lower::IterationSchedule schedule;
  Stmt                     assembleFunc;
  Stmt                     computeFunc;
  shared_ptr<Module>       module;
};

TensorBase::TensorBase() : TensorBase(ComponentType::Double) {
}

TensorBase::TensorBase(ComponentType ctype)
    : TensorBase(util::uniqueName('A'), ctype) {
}

TensorBase::TensorBase(std::string name, ComponentType ctype)
    : TensorBase(name, ctype, {}, Format())  {
}

TensorBase::TensorBase(double val) : TensorBase(type<double>()) {
  this->insert({}, val);
  pack();
}

TensorBase::TensorBase(ComponentType ctype, vector<int> dimensions,
                       Format format)
    : TensorBase(util::uniqueName('A'), ctype, dimensions, format) {
}

TensorBase::TensorBase(string name, ComponentType ctype, vector<int> dimensions,
                       Format format) : content(new Content) {
  taco_uassert(format.getOrder() == dimensions.size() ||
               format.getOrder() == 1) <<
      "The number of format levels (" << format.getOrder() << ") " <<
      "must match the tensor order (" << dimensions.size() << "), " <<
      "or there must be a single level.";
  taco_uassert(ctype == ComponentType::Double) <<
      "Only double tensors currently supported";

  if (dimensions.size() == 0) {
    format = Format();
  }
  else if (dimensions.size() > 1 && format.getOrder() == 1) {
    DimensionType levelType = format.getLevels()[0].getType();
    vector<DimensionType> levelTypes;
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

  // Initialize dense storage dimensions
  vector<Level> levels = format.getLevels();
  for (size_t i=0; i < levels.size(); ++i) {
    if (levels[i].getType() == DimensionType::Dense) {
      auto index = (int*)malloc(sizeof(int));
      index[0] = dimensions[i];
      content->storage.setDimensionIndex(i, {index});
    }
  }
  
  content->module = make_shared<Module>();

  this->coordinateBuffer = shared_ptr<vector<char>>(new vector<char>);
  this->coordinateBufferUsed = 0;
  this->coordinateSize = getOrder()*sizeof(int) + ctype.bytes();
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

const ComponentType& TensorBase::getComponentType() const {
  return content->ctype;
}

const vector<taco::Var>& TensorBase::getIndexVars() const {
  return content->indexVars;
}

const taco::Expr& TensorBase::getExpr() const {
  return content->expr;
}

const storage::Storage& TensorBase::getStorage() const {
  return content->storage;
}

storage::Storage& TensorBase::getStorage() {
  return content->storage;
}

void TensorBase::setAllocSize(size_t allocSize) const {
  taco_uassert(allocSize >= 2 && (allocSize & (allocSize - 1)) == 0) <<
      "The index allocation size must be a power of two and at least two";
  content->allocSize = allocSize;
}

size_t TensorBase::getAllocSize() const {
  return content->allocSize;
}

void TensorBase::setCSR(double* vals, int* rowPtr, int* colIdx) {
  taco_uassert(getFormat() == CSR) <<
      "setCSR: the tensor " << getName() << " is not in the CSR format, " <<
      "but instead " << getFormat();
  auto storage = getStorage();
  storage.setDimensionIndex(0, {util::copyToArray({getDimensions()[0]})});
  storage.setDimensionIndex(1, {rowPtr, colIdx});
  storage.setValues(vals);
}

void TensorBase::getCSR(double** vals, int** rowPtr, int** colIdx) {
  taco_uassert(getFormat() == CSR) <<
      "getCSR: the tensor " << getName() << " is not defined in the CSR format";
  auto storage = getStorage();
  *vals = storage.getValues();
  *rowPtr = storage.getDimensionIndex(1)[0];
  *colIdx = storage.getDimensionIndex(1)[1];
}

void TensorBase::setCSC(double* vals, int* colPtr, int* rowIdx) {
  taco_uassert(getFormat() == CSC) <<
      "setCSC: the tensor " << getName() << " is not defined in the CSC format";
  auto storage = getStorage();
  std::vector<int> denseDim = {getDimensions()[1]};
  storage.setDimensionIndex(0, {util::copyToArray(denseDim)});
  storage.setDimensionIndex(1, {colPtr, rowIdx});
  storage.setValues(vals);
}

void TensorBase::getCSC(double** vals, int** colPtr, int** rowIdx) {
  taco_uassert(getFormat() == CSC) <<
      "getCSC: the tensor " << getName() << " is not defined in the CSC format";

  auto storage = getStorage();
  *vals = storage.getValues();
  *colPtr = storage.getDimensionIndex(1)[0];
  *rowIdx = storage.getDimensionIndex(1)[1];
}

static int numIntegersToCompare = 0;
static int lexicographicalCmp(const void* a, const void* b) {
  for (int i = 0; i < numIntegersToCompare; i++) {
    int diff = ((int*)a)[i] - ((int*)b)[i];
    if (diff != 0) {
      return diff;
    }
  }
  return 0;
}

/// Pack coordinates into a data structure given by the tensor format.
void TensorBase::pack() {
  taco_tassert(getComponentType() == ComponentType::Double)
      << "make the packing machinery work with other primitive types later. "
      << "Right now we're specializing to doubles so that we can use a "
      << "resizable vector, but eventually we should use a two pass pack "
      << "algorithm that figures out sizes first, and then packs the data";

  // Nothing to pack
  if (coordinateBufferUsed == 0) {
    return;
  }
  const size_t order = getOrder();


  // Pack scalars
  if (order == 0) {
    content->storage.setValues((double*)malloc(getComponentType().bytes()));
    char* coordLoc = this->coordinateBuffer->data();
    content->storage.getValues()[0] =
        *(double*)&coordLoc[this->coordinateSize-getComponentType().bytes()];
    this->coordinateBuffer->clear();
    return;
  }


  /// Permute the coordinates according to the storage dimension ordering.
  /// This is a workaround since the current pack code only packs tensors in the
  /// order of the dimensions.
  const std::vector<Level>& levels     = getFormat().getLevels();
  const std::vector<int>&   dimensions = getDimensions();
  taco_iassert(levels.size() == order);
  std::vector<int> permutation;
  for (auto& level : levels) {
    permutation.push_back((int)level.getDimension());
  }

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


  // Move coords into separate arrays
  std::vector<std::vector<int>> coordinates(order);
  for (size_t i=0; i < order; ++i) {
    coordinates[i] = std::vector<int>(numCoordinates);
  }

  std::vector<double> values(numCoordinates);
  for (size_t i=0; i < numCoordinates; ++i) {
    int* coordLoc = (int*)&coordinatesPtr[i*coordSize];
    for (size_t d=0; d < order; ++d) {
      coordinates[d][i] = *coordLoc;
      coordLoc++;
    }
    values[i] = *((double*)coordLoc);
  }
  taco_iassert(coordinates.size() > 0);
  this->coordinateBuffer->clear();
  this->coordinateBufferUsed = 0;


  // Pack indices and values
  content->storage = storage::pack(permutedDimensions, getFormat(),
                                   coordinates, values);

//  ir::Stmt packFunc = storage::packCode(getFormat());
//  std::cout << getFormat() << std::endl;
//  std::cout << packFunc << std::endl;
}

void TensorBase::zero() {
  auto resultStorage = getStorage();
  // Set values to 0.0 in case we are doing a += operation
  memset(resultStorage.getValues(), 0, content->valuesSize * sizeof(double));
}

Access TensorBase::operator()(const std::vector<Var>& indices) {
  taco_uassert(indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " <<
  util::join(indices);
  return Access(*this, indices);
}

void TensorBase::compile() {
  taco_iassert(getExpr().defined()) << "No expression defined for tensor";
  content->assembleFunc = lower::lower(*this, "assemble", {lower::Assemble});
  content->computeFunc  = lower::lower(*this, "compute", {lower::Compute});
  content->module->addFunction(content->assembleFunc);
  content->module->addFunction(content->computeFunc);
  content->module->compile();
}

static inline vector<void*> packArguments(const TensorBase& tensor) {
  vector<void*> arguments;

  // Pack the result tensor
  auto resultStorage = tensor.getStorage();
  auto resultFormat = resultStorage.getFormat();
  for (size_t i=0; i<resultFormat.getLevels().size(); i++) {
    auto& index = resultStorage.getDimensionIndex(i);
    auto& levelFormat = resultFormat.getLevels()[i];
    switch (levelFormat.getType()) {
      case Dense:
        arguments.push_back((void*)index[0]);
        break;
      case Sparse:
      case Fixed:
        arguments.push_back((void*)index[0]);
        arguments.push_back((void*)index[1]);
        break;
    }
  }
  arguments.push_back((void*)resultStorage.getValues());

  // Pack operand tensors
  vector<TensorBase> operands = expr_nodes::getOperands(tensor.getExpr());
  for (auto& operand : operands) {
    Storage storage = operand.getStorage();
    Format format = storage.getFormat();
    for (size_t i=0; i<format.getOrder(); i++) {
      auto& index = storage.getDimensionIndex(i);
      auto& levelFormat = format.getLevels()[i];
      switch (levelFormat.getType()) {
        case Dense:
          arguments.push_back((void*)index[0]);
          break;
        case Sparse:
        case Fixed:
          arguments.push_back((void*)index[0]);
          arguments.push_back((void*)index[1]);
          break;
      }
    }
    arguments.push_back((void*)storage.getValues());
  }

  return arguments;
}

void TensorBase::assemble() {
  this->content->arguments = packArguments(*this);
  this->assembleInternal();
}

void TensorBase::compute() {
  this->content->arguments = packArguments(*this);
  this->zero();
  this->computeInternal();
}

void TensorBase::evaluate() {
  this->compile();
  this->content->arguments = packArguments(*this);
  this->assembleInternal();
  this->zero();
  this->computeInternal();
}

void TensorBase::setExpr(const vector<taco::Var>& indexVars, taco::Expr expr) {
  // The following are index expressions we don't currently support, but that
  // are planned for the future.

  // We don't yet support distributing tensors. That is, every free variable
  // must be used on the right-hand-side.
  set<taco::Var> rhsVars;
  using namespace expr_nodes;
  expr_nodes::match(expr,
    function<void(const ReadNode*)>([&](const ReadNode* op) {
      for (auto& var : op->indexVars) {
        rhsVars.insert(var);
      }
    })
  );
  for (auto& lhsVar : indexVars) {
    taco_uassert(util::contains(rhsVars, lhsVar)) <<
        "All variables must appear on the right-hand-side of an assignment. "
        "This restriction will be removed in the future.\n" <<
        "Expression: " << getName() << "(" << util::join(indexVars,",") << ")"<<
        " = " << expr;
  }

  content->indexVars = indexVars;
  content->expr = expr;

  storage::Storage storage = getStorage();
  Format format = storage.getFormat();
  auto& levels = format.getLevels();
  for (size_t i=0; i < levels.size(); ++i) {
    Level level = levels[i];
    switch (level.getType()) {
      case DimensionType::Dense:
        break;
      case DimensionType::Sparse: {
        auto pos = (int*)malloc(getAllocSize() * sizeof(int));
        auto idx = (int*)malloc(getAllocSize() * sizeof(int));
        pos[0] = 0;
        storage.setDimensionIndex(i, {pos,idx});
        break;
      }
      case DimensionType::Fixed: {
        auto pos = (int*)malloc(sizeof(int));
        auto idx = (int*)malloc(getAllocSize() * sizeof(int));
        storage.setDimensionIndex(i, {pos,idx});
        break;
      }
    }
  }
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
  content->module->setSource(source);
  content->module->compile();
}

bool equals(const TensorBase& a, const TensorBase& b) {
  // Component type must be the same
  if (a.getComponentType() != b.getComponentType()) {
    return false;
  }

  // Dimensions must be the same
  if (a.getOrder() != b.getOrder()) {
    return false;
  }
  for (auto dims : util::zip(a.getDimensions(), b.getDimensions())) {
    if (dims.first != dims.second) {
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

void TensorBase::assembleInternal() {
  content->module->callFunc("assemble", content->arguments.data());
  
  size_t j = 0;
  auto resultStorage = getStorage();
  auto resultFormat = resultStorage.getFormat();
  for (size_t i=0; i<resultFormat.getLevels().size(); i++) {
    auto& levelFormat = resultFormat.getLevels()[i];
    switch (levelFormat.getType()) {
      case Dense:
        j++;
        break;
      case Sparse:
      case Fixed: {
        auto pos = (int*)content->arguments[j++];
        auto idx = (int*)content->arguments[j++];
        resultStorage.setDimensionIndex(i, {pos,idx});
        break;
      }
    }
  }

  content->valuesSize = resultStorage.getSize().numValues();
  resultStorage.setValues((double*)malloc(content->valuesSize*sizeof(double)));
  content->arguments[j] = resultStorage.getValues();
}

void TensorBase::computeInternal() {
  this->content->module->callFunc("compute", content->arguments.data());
}

ostream& operator<<(ostream& os, const TensorBase& tensor) {
  vector<string> dimStrings;
  for (int dim : tensor.getDimensions()) {
    dimStrings.push_back(to_string(dim));
  }
  os << tensor.getName() << " (" << util::join(dimStrings, "x") << ") "
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
    case FileType::dns:
      tensor = io::dns::read(file, format, pack);
      break;
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
  if (extension == "dns") {
    tensor = dispatchRead(filename, FileType::dns, format, pack);
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
    case FileType::dns:
      io::dns::write(file, tensor);
      break;
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
  if (extension == "dns") {
    dispatchWrite(filename, tensor, FileType::dns);
  }
  else if (extension == "tns") {
    dispatchWrite(filename, tensor, FileType::tns);
  }
  else if (extension == "mtx") {
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

void packOperands(const TensorBase& tensor) {
  for (TensorBase operand : expr_nodes::getOperands(tensor.getExpr())) {
    operand.pack();
  }
}

}
