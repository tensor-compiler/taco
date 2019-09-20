#include "taco/tensor.h"

#include <set>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <vector>

#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include <taco/index_notation/transformations.h>
#include "taco/storage/storage.h"
#include "taco/lower/lower.h"
#include "taco/codegen/module.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_cuda.h"
#include "taco/taco_tensor_t.h"
#include "taco/storage/file_io_tns.h"
#include "taco/storage/file_io_mtx.h"
#include "taco/storage/file_io_rb.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"
#include "taco/util/name_generator.h"
#include "taco/error/error_messages.h"
#include "error/error_checks.h"
#include "taco/cuda.h"

using namespace std;
using namespace taco::ir;

namespace taco {

static vector<Dimension> convert(const vector<int>& dimensions) {
  vector<Dimension> dims;
  for (auto& dim : dimensions) {
    dims.push_back(dim);
  }
  return dims;
}

struct TensorBase::Content {
  Datatype           dataType;
  vector<int>        dimensions;

  TensorStorage      storage;
  TensorVar          tensorVar;
  Assignment         assignment;

  size_t             allocSize;
  size_t             valuesSize;

  Stmt               assembleFunc;
  Stmt               computeFunc;
  bool               assembleWhileCompute;
  shared_ptr<Module> module;

  Content(string name, Datatype dataType, const vector<int>& dimensions,
          Format format)
      : dataType(dataType), dimensions(dimensions),
        storage(TensorStorage(dataType, dimensions, format)),
        tensorVar(TensorVar(name, Type(dataType,convert(dimensions)),format)) {}
};

TensorBase::TensorBase() : TensorBase(Float()) {
}

TensorBase::TensorBase(Datatype ctype)
    : TensorBase(util::uniqueName('A'), ctype) {
}

TensorBase::TensorBase(std::string name, Datatype ctype)
    : TensorBase(name, ctype, {}, Format())  {
}

TensorBase::TensorBase(Datatype ctype, vector<int> dimensions, 
                       ModeFormat modeType)
    : TensorBase(util::uniqueName('A'), ctype, dimensions, 
                 std::vector<ModeFormatPack>(dimensions.size(), modeType)) {
}

TensorBase::TensorBase(Datatype ctype, vector<int> dimensions, Format format) 
    : TensorBase(util::uniqueName('A'), ctype, dimensions, format) {
}

TensorBase::TensorBase(std::string name, Datatype ctype, 
                       std::vector<int> dimensions, ModeFormat modeType) 
    : TensorBase(name, ctype, dimensions, 
                 std::vector<ModeFormatPack>(dimensions.size(), modeType)) {
}

static Format initFormat(Format format) {
  // Initialize coordinate types for Format if not already set
  if (format.getLevelArrayTypes().size() < (size_t)format.getOrder()) {
    std::vector<std::vector<Datatype>> levelArrayTypes;
    for (int i = 0; i < format.getOrder(); ++i) {
      std::vector<Datatype> arrayTypes;
      ModeFormat modeType = format.getModeFormats()[i];
      if (modeType.getName() == Dense.getName()) {
        arrayTypes.push_back(Int32);
      } else if (modeType.getName() == Sparse.getName()) {
        arrayTypes.push_back(Int32);
        arrayTypes.push_back(Int32);
      } else if (modeType.getName() == Singleton.getName()) {
        arrayTypes.push_back(Int32);
        arrayTypes.push_back(Int32);
      } else {
        taco_not_supported_yet;
      }
      levelArrayTypes.push_back(arrayTypes);
    }
    format.setLevelArrayTypes(levelArrayTypes);
  }
  return format;
}

TensorBase::TensorBase(string name, Datatype ctype, vector<int> dimensions,
                       Format format)
    : content(new Content(name, ctype, dimensions, initFormat(format))) {
  taco_uassert((size_t)format.getOrder() == dimensions.size()) <<
      "The number of format mode types (" << format.getOrder() << ") " <<
      "must match the tensor order (" << dimensions.size() << ").";

  content->allocSize = 1 << 20;

  vector<ModeIndex> modeIndices(format.getOrder());
  // Initialize dense storage modes
  // TODO: Get rid of this and make code use dimensions instead of dense indices
  for (int i = 0; i < format.getOrder(); ++i) {
    if (format.getModeFormats()[i].getName() == Dense.getName()) {
      const size_t idx = format.getModeOrdering()[i];
      modeIndices[i] = ModeIndex({makeArray({content->dimensions[idx]})});
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
  content->tensorVar.setName(name);
}

string TensorBase::getName() const {
  return content->tensorVar.getName();
}

int TensorBase::getOrder() const {
  return (int)content->dimensions.size();
}

const Format& TensorBase::getFormat() const {
  return content->storage.getFormat();
}

void TensorBase::reserve(size_t numCoordinates) {
  size_t newSize = this->coordinateBuffer->size() +
                   numCoordinates*this->coordinateSize;
  this->coordinateBuffer->resize(newSize);
}

int TensorBase::getDimension(int mode) const {
  taco_uassert(mode < getOrder()) << "Invalid mode";
  return content->dimensions[mode];
}

const vector<int>& TensorBase::getDimensions() const {
  return content->dimensions;
}

const Datatype& TensorBase::getComponentType() const {
  return content->dataType;
}

const TensorVar& TensorBase::getTensorVar() const {
  return content->tensorVar;
}

const TensorStorage& TensorBase::getStorage() const {
  return content->storage;
}

TensorStorage& TensorBase::getStorage() {
  return content->storage;
}

void TensorBase::setAllocSize(size_t allocSize) {
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

static size_t unpackTensorData(const taco_tensor_t& tensorData,
                               const TensorBase& tensor) {
  auto storage = tensor.getStorage();
  auto format = storage.getFormat();

  vector<ModeIndex> modeIndices;
  size_t numVals = 1;
  for (int i = 0; i < tensor.getOrder(); i++) {
    ModeFormat modeType = format.getModeFormats()[i];
    if (modeType.getName() == Dense.getName()) {
      Array size = makeArray({*(int*)tensorData.indices[i][0]});
      modeIndices.push_back(ModeIndex({size}));
      numVals *= ((int*)tensorData.indices[i][0])[0];
    } else if (modeType.getName() == Sparse.getName()) {
      auto size = ((int*)tensorData.indices[i][0])[numVals];
      Array pos = Array(type<int>(), tensorData.indices[i][0], numVals+1, Array::UserOwns);
      Array idx = Array(type<int>(), tensorData.indices[i][1], size, Array::UserOwns);
      modeIndices.push_back(ModeIndex({pos, idx}));
      numVals = size;
    } else if (modeType.getName() == Singleton.getName()) {
      Array idx = Array(type<int>(), tensorData.indices[i][1], numVals, Array::UserOwns);
      modeIndices.push_back(ModeIndex({makeArray(type<int>(), 0), idx}));
    } else {
      taco_not_supported_yet;
    }
  }
  storage.setIndex(Index(format, modeIndices));
  storage.setValues(Array(tensor.getComponentType(), tensorData.vals, numVals));
  return numVals;
}

/// Pack coordinates into a data structure given by the tensor format.
void TensorBase::pack() {
  const int order = getOrder();
  const int csize = getComponentType().getNumBytes();
  const std::vector<int>& dimensions = getDimensions();

  taco_iassert((this->coordinateBufferUsed % this->coordinateSize) == 0);
  const size_t numCoordinates = this->coordinateBufferUsed / this->coordinateSize;
  
  const auto helperFuncs = getHelperFunctions(getFormat(), getComponentType(), 
                                              dimensions);

  // Pack scalars
  if (order == 0) {
    Array array = makeArray(getComponentType(), 1);

    std::vector<taco_mode_t> bufferModeType = {taco_mode_sparse};
    std::vector<int> bufferDim = {1};
    std::vector<int> bufferModeOrdering = {0};
    std::vector<int> bufferCoords(numCoordinates, 0);
    taco_tensor_t* bufferStorage = init_taco_tensor_t(1, csize, 
        (int32_t*)bufferDim.data(), (int32_t*)bufferModeOrdering.data(),
        (taco_mode_t*)bufferModeType.data());
    std::vector<int> pos = {0, (int)numCoordinates};
    bufferStorage->indices[0][0] = (uint8_t*)pos.data();
    bufferStorage->indices[0][1] = (uint8_t*)bufferCoords.data();
    bufferStorage->vals = (uint8_t*)this->coordinateBuffer->data();

    std::vector<void*> arguments = {content->storage, bufferStorage};
    helperFuncs->callFuncPacked("pack", arguments.data());
    content->valuesSize = unpackTensorData(*((taco_tensor_t*)arguments[0]), *this);

    deinit_taco_tensor_t(bufferStorage);
    this->coordinateBuffer->clear();
    return;
  }
    
  // Permute the coordinates according to the storage mode ordering.
  // This is a workaround since the current pack code only packs tensors in the
  // ordering of the modes.
  taco_iassert(getFormat().getOrder() == order);
  std::vector<int> permutation = getFormat().getModeOrdering();
  std::vector<int> permutedDimensions(order);
  for (int i = 0; i < order; ++i) {
    permutedDimensions[i] = dimensions[permutation[i]];
  }
  
  const size_t coordSize = this->coordinateSize;
  char* coordinatesPtr = coordinateBuffer->data();
  vector<int> permuteBuffer(order);
  for (size_t i = 0; i < numCoordinates; ++i) {
    int* coordinate = (int*)coordinatesPtr;
    for (int j = 0; j < order; j++) {
      permuteBuffer[j] = coordinate[permutation[j]];
    }
    for (int j = 0; j < order; j++) {
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
  for (int i = 0; i < order; ++i) {
    coordinates[i] = std::vector<int>(numCoordinates);
  }
  char* values = (char*) malloc(numCoordinates * csize);
  for (size_t i = 0; i < numCoordinates; ++i) {
    int* coordLoc = (int*)&coordinatesPtr[i * coordSize];
    for (int d = 0; d < order; ++d) {
      coordinates[d][i] = *coordLoc;
      coordLoc++;
    }
    memcpy(&values[i * csize], coordLoc, csize);
  }


  this->coordinateBuffer->clear();
  this->coordinateBufferUsed = 0;

  
  std::vector<taco_mode_t> bufferModeTypes(order, taco_mode_sparse);
  taco_tensor_t* bufferStorage = init_taco_tensor_t(order, csize, 
      (int32_t*)dimensions.data(), (int32_t*)permutation.data(),
      (taco_mode_t*)bufferModeTypes.data());
  std::vector<int> pos = {0, (int)numCoordinates};
  bufferStorage->indices[0][0] = (uint8_t*)pos.data();
  for (int i = 0; i < order; ++i) {
    bufferStorage->indices[i][1] = (uint8_t*)coordinates[i].data();
  }
  bufferStorage->vals = (uint8_t*)values;

  // Pack nonzero components into required format
  std::vector<void*> arguments = {content->storage, bufferStorage};
  helperFuncs->callFuncPacked("pack", arguments.data());
  content->valuesSize = unpackTensorData(*((taco_tensor_t*)arguments[0]), *this);

  free(values);
  deinit_taco_tensor_t(bufferStorage);
}

void TensorBase::setStorage(TensorStorage storage) {
  content->storage = storage;
}

void TensorBase::zero() {
  getStorage().getValues().zero();
}

/// Inherits Access and adds a TensorBase object, so that we can retrieve the
/// tensors that was used in an expression when we later want to pack arguments.
struct AccessTensorNode : public AccessNode {
  AccessTensorNode(TensorBase tensor, const std::vector<IndexVar>& indices)
      :  AccessNode(tensor.getTensorVar(), indices), tensor(tensor) {}
  TensorBase tensor;
  virtual void setAssignment(const Assignment& assignment) {
    tensor.setAssignment(assignment);
  }
};

const Access TensorBase::operator()(const std::vector<IndexVar>& indices) const {
  taco_uassert(indices.size() == (size_t)getOrder())
      << "A tensor of order " << getOrder() << " must be indexed with "
      << getOrder() << " variables, but is indexed with:  "
      << util::join(indices);
  return Access(new AccessTensorNode(*this, indices));
}

Access TensorBase::operator()(const std::vector<IndexVar>& indices) {
  taco_uassert(indices.size() == (size_t)getOrder())
      << "A tensor of order " << getOrder() << " must be indexed with "
      << getOrder() << " variables, but is indexed with:  "
      << util::join(indices);
  return Access(new AccessTensorNode(*this, indices));
}

void TensorBase::compile(bool assembleWhileCompute) {
  Assignment assignment = getAssignment();
  taco_uassert(assignment.defined())
      << error::compile_without_expr;

  content->assembleWhileCompute = assembleWhileCompute;

  IndexStmt stmt = makeConcreteNotation(makeReductionNotation(assignment));
  stmt = reorderLoopsTopologically(stmt);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt);
  content->assembleFunc = lower(stmt, "assemble", true, false);
  content->computeFunc = lower(stmt, "compute",  assembleWhileCompute, true);

  content->module->reset();
  content->module->addFunction(content->assembleFunc);
  content->module->addFunction(content->computeFunc);
  content->module->compile();
}

taco_tensor_t* TensorBase::getTacoTensorT() {
  return getStorage();
}

static inline map<TensorVar, TensorBase> getTensors(const IndexExpr& expr) {
  struct GetOperands : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    set<TensorBase> inserted;
    vector<TensorBase> operands;

    map<TensorVar, TensorBase> arguments;
    void visit(const AccessNode* node) {
      if (!util::contains(arguments, node->tensorVar)) {
        arguments.insert({node->tensorVar, to<AccessTensorNode>(node)->tensor});
      }

      taco_iassert(isa<AccessTensorNode>(node)) << "Unknown subexpression";
      TensorBase tensor = to<AccessTensorNode>(node)->tensor;
      if (!util::contains(inserted, tensor)) {
        inserted.insert(tensor);
        operands.push_back(tensor);
      }
    }
  };
  GetOperands getOperands;
  expr.accept(&getOperands);
  return getOperands.arguments;
}

static inline
vector<void*> packArguments(const TensorBase& tensor) {
  vector<void*> arguments;

  // Pack the result tensor
  arguments.push_back(tensor.getStorage());

  // Pack operand tensors
  auto operands = getArguments(makeConcreteNotation(tensor.getAssignment()));

  auto tensors = getTensors(tensor.getAssignment().getRhs());
  for (auto& operand : operands) {
    taco_iassert(util::contains(tensors, operand));
    arguments.push_back(tensors.at(operand).getStorage());
  }

  return arguments;
}

void TensorBase::assemble() {
  taco_uassert(this->content->assembleFunc.defined())
      << error::assemble_without_compile;

  auto arguments = packArguments(*this);
  content->module->callFuncPacked("assemble", arguments.data());

  if (!content->assembleWhileCompute) {
    taco_tensor_t* tensorData = ((taco_tensor_t*)arguments[0]);
    content->valuesSize = unpackTensorData(*tensorData, *this);
  }
}

void TensorBase::compute() {
  taco_uassert(this->content->computeFunc.defined())
      << error::compute_without_compile;

  auto arguments = packArguments(*this);
  this->content->module->callFuncPacked("compute", arguments.data());

  if (content->assembleWhileCompute) {
    taco_tensor_t* tensorData = ((taco_tensor_t*)arguments[0]);
    content->valuesSize = unpackTensorData(*tensorData, *this);
  }
}

void TensorBase::evaluate() {
  this->compile();
  if (!getAssignment().getOperator().defined()) {
    this->assemble();
  }
  this->compute();
}

void TensorBase::operator=(const IndexExpr& expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  setAssignment(Assignment(getTensorVar(), {}, expr));
}

void TensorBase::setAssignment(Assignment assignment) {
  content->assignment = makeReductionNotation(assignment);
}

Assignment TensorBase::getAssignment() const {
  return content->assignment;
}

void TensorBase::printComputeIR(ostream& os, bool color, bool simplify) const {
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(os, ir::CodeGen::ImplementationGen);
  codegen->compile(content->computeFunc.as<Function>(), false);
}

void TensorBase::printAssembleIR(ostream& os, bool color, bool simplify) const {
  IRPrinter printer(os, color, simplify);
  printer.print(content->assembleFunc.as<Function>()->body);
}

string TensorBase::getSource() const {
  return content->module->getSource();
}

void TensorBase::compileSource(std::string source) {
  taco_iassert(getAssignment().getRhs().defined())
      << error::compile_without_expr;

  IndexStmt stmt = makeConcreteNotation(makeReductionNotation(getAssignment()));
  stmt = reorderLoopsTopologically(stmt);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt);
  content->assembleFunc = lower(stmt, "assemble", true, false);
  content->computeFunc = lower(stmt, "compute",  false, true);

  stringstream ss;
  if (should_use_CUDA_codegen()) {
    CodeGen_CUDA::generateShim(content->assembleFunc, ss);
    ss << endl;
    CodeGen_CUDA::generateShim(content->computeFunc, ss);
  }
  else {
    CodeGen_C::generateShim(content->assembleFunc, ss);
    ss << endl;
    CodeGen_C::generateShim(content->computeFunc, ss);
  }
  content->module->setSource(source + "\n" + ss.str());
  content->module->compile();
}

std::vector<std::tuple<Format,
                       Datatype,
                       std::vector<int>,
                       std::shared_ptr<Module>>> 
TensorBase::helperFunctions = 
    std::vector<std::tuple<Format,
                           Datatype,
                           std::vector<int>,
                           std::shared_ptr<Module>>>();

std::shared_ptr<ir::Module> 
TensorBase::getHelperFunctions(const Format& format, Datatype ctype, 
                               const std::vector<int>& dimensions) {
  for (const auto& helperFunctions : TensorBase::helperFunctions) {
    if (std::get<0>(helperFunctions) == format &&
        std::get<1>(helperFunctions) == ctype && 
        std::get<2>(helperFunctions) == dimensions) {
      // If helper functions had already been generated for specified tensor 
      // format and type, then use cached version.
      return std::get<3>(helperFunctions);
    }
  }
  
  std::shared_ptr<Module> helperModule = std::make_shared<Module>();
  helperFunctions.emplace_back(format, ctype, dimensions, helperModule);
  
  if (format.getOrder() > 0) {
    const Format bufferFormat = COO(format.getOrder(), false, true, false, 
                                    format.getModeOrdering());
    TensorBase bufferTensor(ctype, dimensions, bufferFormat);
    TensorBase packedTensor(ctype, dimensions, format);

    // Define packing and iterator routines in index notation.
    std::vector<IndexVar> indexVars(format.getOrder());
    IndexStmt packStmt = (packedTensor(indexVars) = bufferTensor(indexVars));
    IndexStmt iterateStmt = Yield(indexVars, packedTensor(indexVars));
    for (int i = format.getOrder() - 1; i >= 0; --i) {
      int mode = format.getModeOrdering()[i];
      packStmt = forall(indexVars[mode], packStmt);
      iterateStmt = forall(indexVars[mode], iterateStmt);
    }

    // Lower packing and iterator code.
    helperModule->addFunction(lower(packStmt, "pack", true, true));
    helperModule->addFunction(lower(iterateStmt, "iterate", false, true));
  } else {
    const Format bufferFormat = COO(1, false, true, false);
    TensorBase bufferVector(ctype, {1}, bufferFormat);
    TensorBase packedScalar(ctype, dimensions, format);

    // Define and lower packing routine.
    // TODO: Redefine as reduction into packed scalar once reduction bug 
    //       has been fixed in new lowering machinery.
    IndexVar indexVar;
    IndexStmt assignment = (packedScalar() = bufferVector(indexVar));
    IndexStmt packStmt= makeConcreteNotation(makeReductionNotation(assignment));
    helperModule->addFunction(lower(packStmt, "pack", true, true));

    // Define and lower iterator code.
    IndexStmt iterateStmt = Yield({}, packedScalar());
    helperModule->addFunction(lower(iterateStmt, "iterate", false, true));
  }
  helperModule->compile();

  return helperModule;
}

template<typename T>
bool isZero(T a) {
  if ((double)a == 0.0) {
    return true;
  }
  return false;
}

template<typename T>
bool isZero(std::complex<T> a) {
  if (a.real() == 0.0 && a.imag() == 0.0) {
    return true;
  }
  return false;
}

template<typename T>
bool scalarEquals(T a, T b) {
  double diff = ((double) a - (double) b)/(double)a;
  if (std::abs(diff) > 10e-6) {
    return false;
  }
  return true;
}

template<typename T>
bool scalarEquals(std::complex<T> a, std::complex<T> b) {
  T diff = std::abs((a - b)/a);
  if ((diff > 10e-6) || (diff < -10e-6)) {
    return false;
  }
  return true;
}
  
template<typename T>
bool equalsTyped(const TensorBase& a, const TensorBase& b) {
  auto at = iterate<T>(a);
  auto bt = iterate<T>(b);
  auto ait = at.begin();
  auto bit = bt.begin();
  
  while (ait != at.end() && bit != bt.end()) {
    auto acoord = ait->first;
    auto bcoord = bit->first;
    auto aval = ait->second;
    auto bval = bit->second;

    if (acoord != bcoord) {
      if (isZero(aval)) {
        ++ait;
        continue;
      }
      else if (isZero(bval)) {
        ++bit;
        continue;
      }

      return false;
    }
    if (!scalarEquals(aval, bval)) {
      return false;
    }

    ++ait;
    ++bit;
  }
  while (ait != at.end()) {
    auto aval = ait->second;
    if (!isZero(aval)) {
      return false;
    }
    ++ait;
  }
  while (bit != bt.end()) {
    auto bval = bit->second;
    if (!isZero(bval)) {
      return false;
    }
    ++bit;
  }
  return (ait == at.end() && bit == bt.end());
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
  for (int mode = 0; mode < a.getOrder(); mode++) {
    if (a.getDimension(mode) != b.getDimension(mode)) {
      return false;
    }
  }

  // Values must be the same
  switch(a.getComponentType().getKind()) {
    case Datatype::Bool: taco_ierror; return false;
    case Datatype::UInt8: return equalsTyped<uint8_t>(a, b);
    case Datatype::UInt16: return equalsTyped<uint16_t>(a, b);
    case Datatype::UInt32: return equalsTyped<uint32_t>(a, b);
    case Datatype::UInt64: return equalsTyped<uint64_t>(a, b);
    case Datatype::UInt128: return equalsTyped<unsigned long long>(a, b);
    case Datatype::Int8: return equalsTyped<int8_t>(a, b);
    case Datatype::Int16: return equalsTyped<int16_t>(a, b);
    case Datatype::Int32: return equalsTyped<int32_t>(a, b);
    case Datatype::Int64: return equalsTyped<int64_t>(a, b);
    case Datatype::Int128: return equalsTyped<long long>(a, b);
    case Datatype::Float32: return equalsTyped<float>(a, b);
    case Datatype::Float64: return equalsTyped<double>(a, b);
    case Datatype::Complex64: return equalsTyped<std::complex<float>>(a, b);
    case Datatype::Complex128: return equalsTyped<std::complex<double>>(a, b);
    case Datatype::Undefined: taco_ierror << "Undefined data type"; 
  }
  taco_unreachable;
  return false;
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
    os << "(" << util::join(ptr, ptr+tensor.getOrder()) << "): ";
    switch(tensor.getComponentType().getKind()) {
      case Datatype::Bool: taco_ierror; break;
      case Datatype::UInt8: os << ((uint8_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::UInt16: os << ((uint16_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::UInt32: os << ((uint32_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::UInt64: os << ((uint64_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::UInt128: os << ((unsigned long long*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Int8: os << ((int8_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Int16: os << ((int16_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Int32: os << ((int32_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Int64: os << ((int64_t*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Int128: os << ((long long*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Float32: os << ((float*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Float64: os << ((double*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Complex64: os << ((std::complex<float>*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Complex128: os << ((std::complex<double>*)(ptr+tensor.getOrder()))[0] << std::endl; break;
      case Datatype::Undefined: taco_ierror; break;
    }
  }

  // Print packed data
  os << tensor.getStorage();

  return os;
}

static string getExtension(string filename) {
  return filename.substr(filename.find_last_of(".") + 1);
}

template <typename T, typename U>
TensorBase dispatchRead(T& file, FileType filetype, U format, bool pack) {
  TensorBase tensor;
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      tensor = readMTX(file, format, pack);
      break;
    case FileType::tns:
      tensor = readTNS(file, format, pack);
      break;
    case FileType::rb:
      tensor = readRB(file, format, pack);
      break;
  }
  return tensor;
}

template <typename U>
TensorBase dispatchRead(std::string filename, U format, bool pack) {
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

TensorBase read(std::string filename, ModeFormat modetype, bool pack) {
  return dispatchRead(filename, modetype, pack);
}

TensorBase read(std::string filename, Format format, bool pack) {
  return dispatchRead(filename, format, pack);
}

TensorBase read(string filename, FileType filetype, ModeFormat modetype, 
                bool pack) {
  return dispatchRead(filename, filetype, modetype, pack);
}

TensorBase read(string filename, FileType filetype, Format format, bool pack) {
  return dispatchRead(filename, filetype, format, pack);
}

TensorBase read(istream& stream, FileType filetype, ModeFormat modetype, 
                bool pack) {
  return dispatchRead(stream, filetype, modetype, pack);
}

TensorBase read(istream& stream, FileType filetype, Format format, bool pack) {
  return dispatchRead(stream, filetype, format, pack);
}

template <typename T>
void dispatchWrite(T& file, const TensorBase& tensor, FileType filetype) {
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      writeMTX(file, tensor);
      break;
    case FileType::tns:
      writeTNS(file, tensor);
      break;
    case FileType::rb:
      writeRB(file, tensor);
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

void packOperands(const TensorBase& tensor) {
  auto operands = getArguments(makeConcreteNotation(tensor.getAssignment()));
  auto tensors = getTensors(tensor.getAssignment().getRhs());
  for (auto& operand : operands) {
    taco_iassert(util::contains(tensors, operand)) << operand.getName();
    tensors.at(operand).pack();
  }
}

static ParallelSchedule taco_parallel_sched = ParallelSchedule::Static;
static int taco_chunk_size = 0;
static int taco_num_threads = 1;

void taco_set_parallel_schedule(ParallelSchedule sched, int chunk_size) {
  taco_parallel_sched = sched;
  taco_chunk_size = chunk_size;
}

void taco_get_parallel_schedule(ParallelSchedule *sched, int *chunk_size) {
  *sched = taco_parallel_sched;
  *chunk_size = taco_chunk_size;
}

void taco_set_num_threads(int num_threads) {
  if (num_threads > 0) {
    taco_num_threads = num_threads;
  }
}

int taco_get_num_threads() {
  return taco_num_threads;
}

}
