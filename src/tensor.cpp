#include "taco/tensor.h"

#include <set>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <climits>
#include <vector>
#include <utility>
#include <mutex>

#include "taco/cuda.h"
#include "taco/format.h"
#include "taco/taco_tensor_t.h"
#include "taco/codegen/module.h"
#include "taco/error/error_messages.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
//#include "codegen/codegen_c.h"
//#include "codegen/codegen_cuda.h"
//#include "taco/taco_tensor_t.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/transformations.h"
#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "taco/lower/lower.h"
#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/pack.h"
#include "taco/storage/file_io_tns.h"
#include "taco/storage/file_io_mtx.h"
#include "taco/storage/file_io_rb.h"
#include "taco/storage/typed_vector.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"
#include "taco/util/name_generator.h"

#include "codegen/codegen_c.h"
#include "codegen/codegen_cuda.h"
#include "error/error_checks.h"
#include "taco/cuda.h"
#include "lower/iteration_graph.h"

using namespace std;
using namespace taco::ir;

namespace taco {

TensorBase::TensorBase() : TensorBase(Float()) {
}

TensorBase::TensorBase(Datatype ctype)
    : TensorBase(util::uniqueName('A'), ctype) {
}

TensorBase::TensorBase(std::string name, Datatype ctype)
    : TensorBase(name, ctype, {}, Format(), Literal::zero(ctype))  {
}

TensorBase::TensorBase(Datatype ctype, vector<int> dimensions, 
                       ModeFormat modeType, Literal fill)
    : TensorBase(util::uniqueName('A'), ctype, dimensions, 
                 std::vector<ModeFormatPack>(dimensions.size(), modeType), fill) {
}

TensorBase::TensorBase(Datatype ctype, vector<int> dimensions, Format format, Literal fill)
    : TensorBase(util::uniqueName('A'), ctype, dimensions, format, fill) {
}

TensorBase::TensorBase(std::string name, Datatype ctype, 
                       std::vector<int> dimensions, ModeFormat modeType, Literal fill)
    : TensorBase(name, ctype, dimensions, 
                 std::vector<ModeFormatPack>(dimensions.size(), modeType), fill) {
}

TensorBase::TensorBase(Datatype ctype, std::vector<int> dimensions, Literal fill)
    : TensorBase(ctype, dimensions, ModeFormat::compressed, fill) {
}

TensorBase::TensorBase(std::string name, Datatype ctype, std::vector<int> dimensions, Literal fill)
    : TensorBase(name, ctype, dimensions, ModeFormat::compressed, fill) {
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
                       Format format, Literal fill) {

  // Default fill to zero since undefined. This is done since we need the ctype to initialize the
  // fill and we can't use this inside the default arguments.
  fill = fill.defined()? fill : Literal::zero(ctype);
  content = shared_ptr<Content>(new Content(name, ctype, dimensions, initFormat(format), fill));

  taco_uassert((size_t)format.getOrder() == dimensions.size()) <<
      "The number of format mode types (" << format.getOrder() << ") " <<
      "must match the tensor order (" << dimensions.size() << ").";

  taco_uassert(ctype == fill.getDataType()) << "Fill value must be of the same type as the tensor.";

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

  content->neverPacked = true;
  content->needsPack = true;
  content->needsCompile = false;
  content->needsAssemble = false;
  content->needsCompute = false;

  content->coordinateBuffer = shared_ptr<vector<char>>(new vector<char>);
  content->coordinateBufferUsed = 0;
  content->coordinateSize = getOrder()*sizeof(int) + ctype.getNumBytes();
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
  size_t newSize = content->coordinateBuffer->size() +
                   numCoordinates * content->coordinateSize;
  content->coordinateBuffer->resize(newSize);
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

void TensorBase::unsetNeverPacked() {
  content->neverPacked = false;
}

void TensorBase::setNeedsPack(bool needsPack) {
  content->needsPack = needsPack;
}

void TensorBase::setNeedsCompile(bool needsCompile) {
  content->needsCompile = needsCompile;
}

void TensorBase::setNeedsAssemble(bool needsAssemble) {
  content->needsAssemble = needsAssemble;
}

void TensorBase::setNeedsCompute(bool needsCompute) {
  content->needsCompute = needsCompute;
}

bool TensorBase::neverPacked() {
  return content->neverPacked;
}

bool TensorBase::needsPack() {
  return content->needsPack;
}

bool TensorBase::needsCompile() {
  return content->needsCompile;
}

bool TensorBase::needsAssemble() {
  return content->needsAssemble;
}

bool TensorBase::needsCompute() {
  return content->needsCompute;
}

void TensorBase::setAssembleWhileCompute(bool assembleWhileCompute) {
  content->assembleWhileCompute = assembleWhileCompute;
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
  if (!needsPack()) {
    return;
  }
  setNeedsPack(false);

  if (neverPacked()) {
    unsetNeverPacked();
  } else {
    // Reinsert packed components into temporary buffer and repack them along
    // with unpacked components. This is needed to implement increment
    // semantics.
    // TODO: Change to using code that adds packed components (stored in packed
    //       data structure) with unpacked components (stored in temporary
    //       buffer). We can already generate such code, but currently
    //       compiling it is too expensive.
    switch (getComponentType().getKind()) {
      case Datatype::Bool:
        reinsertPackedComponents<bool>();
        break;
      case Datatype::UInt8:
        reinsertPackedComponents<uint8_t>();
        break;
      case Datatype::UInt16:
        reinsertPackedComponents<uint16_t>();
        break;
      case Datatype::UInt32:
        reinsertPackedComponents<uint32_t>();
        break;
      case Datatype::UInt64:
        reinsertPackedComponents<uint64_t>();
        break;
      case Datatype::Int8:
        reinsertPackedComponents<int8_t>();
        break;
      case Datatype::Int16:
        reinsertPackedComponents<int16_t>();
        break;
      case Datatype::Int32:
        reinsertPackedComponents<int32_t>();
        break;
      case Datatype::Int64:
        reinsertPackedComponents<int64_t>();
        break;
      case Datatype::Float32:
        reinsertPackedComponents<float>();
        break;
      case Datatype::Float64:
        reinsertPackedComponents<double>();
        break;
      case Datatype::Complex64:
        reinsertPackedComponents<std::complex<float>>();
        break;
      case Datatype::Complex128:
        reinsertPackedComponents<std::complex<double>>();
        break;
      default:
        taco_ierror << "unsupported type";
        break;
    };
  }

  const int order = getOrder();
  const int csize = getComponentType().getNumBytes();
  const std::vector<int>& dimensions = getDimensions();

  taco_iassert((content->coordinateBufferUsed % content->coordinateSize) == 0);
  const size_t numCoordinates = content->coordinateBufferUsed / content->coordinateSize;

  const auto helperFuncs = getHelperFunctions(getFormat(), getComponentType(),
                                              dimensions);

  // Pack scalars
  if (order == 0) {
    Array array = makeArray(getComponentType(), 1);

    std::vector<taco_mode_t> bufferModeType = {taco_mode_sparse};
    std::vector<int> bufferDim = {1};
    std::vector<int> bufferModeOrdering = {0};
    std::vector<int> bufferCoords(numCoordinates, 0);

    void* fillPtr = getStorage().getFillValue().defined()? getStorage().getFillValue().getValPtr() : nullptr;
    taco_tensor_t* bufferStorage = init_taco_tensor_t(1, csize,
        (int32_t*)bufferDim.data(), (int32_t*)bufferModeOrdering.data(),
        (taco_mode_t*)bufferModeType.data(), fillPtr);
    std::vector<int> pos = {0, (int)numCoordinates};
    bufferStorage->indices[0][0] = (uint8_t*)pos.data();
    bufferStorage->indices[0][1] = (uint8_t*)bufferCoords.data();

    bufferStorage->vals = (uint8_t*)content->coordinateBuffer->data();

    std::vector<void*> arguments = {content->storage, bufferStorage};
    helperFuncs->callFuncPacked("pack", arguments.data());
    content->valuesSize = unpackTensorData(*((taco_tensor_t*)arguments[0]), *this);

    deinit_taco_tensor_t(bufferStorage);
    content->coordinateBuffer->clear();
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

  const size_t coordSize = content->coordinateSize;
  char* coordinatesPtr = content->coordinateBuffer->data();
  vector<int> permuteBuffer(order);
  for (size_t i = 0; i < numCoordinates; ++i) {
    int* coordinate = (int*)coordinatesPtr;
    for (int j = 0; j < order; j++) {
      permuteBuffer[j] = coordinate[permutation[j]];
    }
    for (int j = 0; j < order; j++) {
      coordinate[j] = permuteBuffer[j];
    }
    coordinatesPtr += content->coordinateSize;
  }
  coordinatesPtr = content->coordinateBuffer->data();

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


  content->coordinateBuffer->clear();
  content->coordinateBufferUsed = 0;

  void* fillPtr = getStorage().getFillValue().defined()? getStorage().getFillValue().getValPtr() : nullptr;
  std::vector<taco_mode_t> bufferModeTypes(order, taco_mode_sparse);
  taco_tensor_t* bufferStorage = init_taco_tensor_t(order, csize,
      (int32_t*)dimensions.data(), (int32_t*)permutation.data(),
      (taco_mode_t*)bufferModeTypes.data(), fillPtr);
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
  // TODO(pnoyola): figure out all possible interactions between
  // setStorage and automatic compilation machinery.
  content->needsPack = false;
  content->storage = storage;
}

static inline map<TensorVar, TensorBase> getTensors(const IndexExpr& expr);

/// Inherits Access and adds a TensorBase object, so that we can retrieve the
/// tensors that was used in an expression when we later want to pack arguments.
struct AccessTensorNode : public AccessNode {
  AccessTensorNode(TensorBase tensor, const std::vector<IndexVar>& indices)
      :  AccessNode(tensor.getTensorVar(), indices, {}, false), 
         tensorPtr(tensor.content) {}

  AccessTensorNode(TensorBase tensor, const std::vector<std::shared_ptr<IndexVarInterface>>& indices)
    : AccessNode(tensor.getTensorVar()), tensorPtr(tensor.content) {
    // Create the vector of IndexVar to assign to this->indexVars.
    std::vector<IndexVar> ivars(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
      auto var = indices[i];
      // Match on what the IndexVarInterface actually is.
      IndexVarInterface::match(var, [&](std::shared_ptr<IndexVar> ivar) {
        ivars[i] = *ivar;
      }, [&](std::shared_ptr<WindowedIndexVar> wvar) {
        ivars[i] = wvar->getIndexVar();
        auto lo = wvar->getLowerBound();
        auto hi = wvar->getUpperBound();
        taco_uassert(lo >= 0) << "slice lower bound must be >= 0";
        taco_uassert(hi <= tensor.getDimension(i)) <<
          "slice upper bound must be <= tensor dimension (" << tensor.getDimension(i) << ")";
        this->windowedModes[i].lo = lo;
        this->windowedModes[i].hi = hi;
        this->windowedModes[i].stride = wvar->getStride();
      }, [&](std::shared_ptr<IndexSetVar> svar) {
        ivars[i] = svar->getIndexVar();
        // Extract the user provided index set.
        auto indexSet = svar->getIndexSet();
        // Ensure that it has at most dim(t, i) elements.
        taco_uassert(indexSet.size() <= size_t(tensor.getDimension(i)));
        // Pack up the index set into a sparse tensor.
        TensorBase indexSetTensor(type<int>(), {int(indexSet.size())}, Compressed);
        for (auto& coord : indexSet) {
          indexSetTensor.insert({coord}, 1);
        }
        indexSetTensor.pack();
        this->indexSetModes[i].set = std::make_shared<std::vector<int>>(indexSet);
        this->indexSetModes[i].tensor = indexSetTensor;
      });
    }
    // Initialize this->indexVars.
    this->indexVars = std::move(ivars);
  }

  // We hold a weak_ptr to the accessed TensorBase to avoid creating a reference
  // cycle between the accessed TensorBase and this AccessTensorNode, since the
  // TensorBase will store the AccessTensorNode (as part of an IndexExpr) as a
  // field on itself. Not using a weak pointer results in leaking TensorBases.
  std::weak_ptr<TensorBase::Content> tensorPtr;
  TensorBase getTensor() const {
    TensorBase tensor;
    tensor.content = tensorPtr.lock();
    return tensor;
  }

  virtual void setAssignment(const Assignment& assignment) {
    auto tensor = this->getTensor();

    tensor.syncDependentTensors();
    Assignment assign = makeReductionNotation(assignment);

    tensor.setNeedsPack(false);
    if (!equals(tensor.getAssignment(), assign)) {
      if (tensor.needsCompute()) {
        auto oldOperands = getTensors(tensor.getAssignment().getRhs());
        for (auto& operand : oldOperands) {
          operand.second.removeDependentTensor(tensor);
        }
      }
      tensor.setNeedsCompile(true);
    }
    tensor.setNeedsAssemble(true);
    tensor.setNeedsCompute(true);

    auto operands = getTensors(assignment.getRhs());
    for (auto& operand : operands) {
      operand.second.addDependentTensor(tensor);
    }

    tensor.setAssignment(assign);
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

Access TensorBase::operator()(const std::vector<std::shared_ptr<IndexVarInterface>>& indices) {
  taco_uassert(indices.size() == (size_t)getOrder())
      << "A tensor of order " << getOrder() << " must be indexed with "
      << getOrder() << " variables, but is indexed with:  "
      << util::join(indices);
  return Access(new AccessTensorNode(*this, indices));
}

Access TensorBase::operator()() {
  return this->operator()(std::vector<IndexVar>());
}

const Access TensorBase::operator()() const {
  return this->operator()(std::vector<IndexVar>());
}

TensorBase::KernelsCache TensorBase::computeKernels;
std::mutex TensorBase::computeKernelsMutex;

std::shared_ptr<Module> TensorBase::getComputeKernel(const IndexStmt stmt) {
  computeKernelsMutex.lock();
  const auto computeKernelsReverse =
      util::ReverseConstIterable<TensorBase::KernelsCache>(computeKernels);
  for (const auto& computeKernel : computeKernelsReverse) {
    if (isomorphic(stmt, computeKernel.first)) {
      const auto kernelModule = computeKernel.second;
      computeKernelsMutex.unlock();
      return kernelModule;
    }
  }
  computeKernelsMutex.unlock();
  return nullptr;
}

void TensorBase::cacheComputeKernel(const IndexStmt stmt,
                                    const std::shared_ptr<Module> kernel) {
  computeKernelsMutex.lock();
  computeKernels.emplace_back(stmt, kernel);
  computeKernelsMutex.unlock();
}

void TensorBase::compile() {
  Assignment assignment = getAssignment();
  taco_uassert(assignment.defined())
      << error::compile_without_expr;

  struct CollisionFinder : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    std::map<std::string,const TensorVar> tensorvars;

    CollisionFinder() :tensorvars() {}

    void visit(const AccessNode* node) {
      Access access(node);
      const TensorVar new_tensorvar = access.getTensorVar();
      const std::string new_name = new_tensorvar.getName();
      if(new_tensorvar.getId() != -1) {
        auto found = tensorvars.find(new_name);
        if(found != tensorvars.end() && found->second.getId() != -1) {
          const TensorVar found_tensorvar = found->second;
          taco_uassert(new_tensorvar.getId() == found_tensorvar.getId())
              << error::compile_tensor_name_collision << " " << new_name;
        } else {
          tensorvars.insert(std::pair<std::string,const TensorVar>(new_name, new_tensorvar));
        }
      }
    }
  };
  CollisionFinder dupes = CollisionFinder();
  assignment.getLhs().accept(&dupes);
  assignment.accept(&dupes);

  IndexStmt stmt = makeConcreteNotation(makeReductionNotation(assignment));
  stmt = reorderLoopsTopologically(stmt);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt);
  compile(stmt, content->assembleWhileCompute);
}

void TensorBase::compile(taco::IndexStmt stmt, bool assembleWhileCompute) {
  if (!needsCompile()) {
    return;
  }
  setNeedsCompile(false);

  IndexStmt concretizedAssign = stmt;
  IndexStmt stmtToCompile = stmt.concretize();
  stmtToCompile = scalarPromote(stmtToCompile);

  if (!std::getenv("CACHE_KERNELS") ||
      std::string(std::getenv("CACHE_KERNELS")) != "0") {
    concretizedAssign = stmtToCompile;
    const auto cachedKernel = getComputeKernel(concretizedAssign);
    if (cachedKernel) {
      content->module = cachedKernel;
      return;
    }
  }

  content->assembleFunc = lower(stmtToCompile, "assemble", true, false);
  content->computeFunc = lower(stmtToCompile, "compute",  assembleWhileCompute, true);
  // If we have to recompile the kernel, we need to create a new Module. Since
  // the module we are holding on to could have been retrieved from the cache,
  // we can't modify it.
  content->module = make_shared<Module>();
  content->module->addFunction(content->assembleFunc);
  content->module->addFunction(content->computeFunc);
  content->module->compile();
  cacheComputeKernel(concretizedAssign, content->module);
}

taco_tensor_t* TensorBase::getTacoTensorT() {
  return getStorage();
}


Literal TensorBase::getFillValue() const {
  return content->tensorVar.getFill();
}

void TensorBase::syncValues() {
  if (content->needsPack) {
    pack();
  } else if (content->needsCompute) {
    compile();
    assemble();
    compute();
  }
}

void TensorBase::addDependentTensor(TensorBase& tensor) {
  content->dependentTensors.push_back(tensor.content);
}

void TensorBase::removeDependentTensor(TensorBase& tensor) {
  int size = content->dependentTensors.size();
  if (size == 0) {
    return;
  }
  TensorBase back;
  back.content = content->dependentTensors[size - 1].lock();

  if (back == tensor) {
    content->dependentTensors.pop_back();
    return;
  }
  for (int i = 0; i < size - 1; i++) {
    TensorBase current;
    current.content = content->dependentTensors[i].lock();
    if (current == tensor) {
      content->dependentTensors[i] = content->dependentTensors[size - 1];
      content->dependentTensors.pop_back();
      return;
    }
  }
}

vector<TensorBase> TensorBase::getDependentTensors() {
  vector<TensorBase> dependents;
  for(std::weak_ptr<Content> dependentContent : content->dependentTensors) {
    TensorBase current;
    current.content = dependentContent.lock();
    dependents.push_back(current);
  }
  return dependents;
}

void TensorBase::syncDependentTensors() {
  vector<TensorBase> dependents = getDependentTensors();
  for (TensorBase dependent : dependents) {
    dependent.syncValues();
  }
  content->dependentTensors.clear();
}

static inline map<TensorVar, TensorBase> getTensors(const IndexExpr& expr) {
  struct GetOperands : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    set<TensorBase> inserted;
    vector<TensorBase> operands;

    map<TensorVar, TensorBase> arguments;
    void visit(const AccessNode* node) {
      if (!isa<AccessTensorNode>(node)) {
        return; // temporary ignore
      }
      taco_iassert(isa<AccessTensorNode>(node)) << "Unknown subexpression";

      if (!util::contains(arguments, node->tensorVar)) {
        arguments.insert({node->tensorVar, to<AccessTensorNode>(node)->getTensor()});
      }

      // Also add any tensors backing index sets of tensor accesses.
      for (auto& p : node->indexSetModes) {
        auto tv = p.second.tensor.getTensorVar();
        if (!util::contains(arguments, tv)) {
          arguments.insert({tv, p.second.tensor});
        }
      }

      // TODO (rohany): This seems like dead code.
      TensorBase tensor = to<AccessTensorNode>(node)->getTensor();
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

  // Pack any index sets on the result tensor at the front of the arguments list.
  auto lhs = getNode(tensor.getAssignment().getLhs());
  // We check isa<AccessNode> rather than isa<AccessTensorNode> to catch cases
  // where the underlying access is represented with the base AccessNode class.
  if (isa<AccessNode>(lhs)) {
    auto indexSetModes = to<AccessNode>(lhs)->indexSetModes;
    for (auto& it : indexSetModes) {
      arguments.push_back(it.second.tensor.getStorage());
    }
  }

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
  taco_uassert(!needsCompile()) << error::assemble_without_compile;
  if (!needsAssemble()) {
    return;
  }
  // Sync operand tensors if needed.
  auto operands = getTensors(getAssignment().getRhs());
  for (auto& operand : operands) {
    operand.second.syncValues();
  }

  auto arguments = packArguments(*this);
  content->module->callFuncPacked("assemble", arguments.data());

  if (!content->assembleWhileCompute) {
    setNeedsAssemble(false);
    taco_tensor_t* tensorData = ((taco_tensor_t*)arguments[0]);
    content->valuesSize = unpackTensorData(*tensorData, *this);
  }
}

void TensorBase::compute() {
  taco_uassert(!needsCompile()) << error::compute_without_compile;
  if (!needsCompute()) {
    return;
  }
  setNeedsCompute(false);
  // Sync operand tensors if needed.
  auto operands = getTensors(getAssignment().getRhs());
  for (auto& operand : operands) {
    operand.second.syncValues();
    operand.second.removeDependentTensor(*this);
  }

  auto arguments = packArguments(*this);
  this->content->module->callFuncPacked("compute", arguments.data());

  if (content->assembleWhileCompute) {
    setNeedsAssemble(false);
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

  syncDependentTensors();
  auto operands = getTensors(expr);
  for (auto& operand : operands) {
    operand.second.addDependentTensor(*this);
  }

  Assignment assign = makeReductionNotation(Assignment(getTensorVar(), {}, expr));

  setNeedsPack(false);
  if (!equals(getAssignment(), assign)) {
    setNeedsCompile(true);
  }
  setNeedsAssemble(true);
  setNeedsCompute(true);

  setAssignment(assign);
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
  setNeedsCompile(false);
}

TensorBase::HelperFuncsCache TensorBase::helperFunctions;
std::mutex TensorBase::helperFunctionsMutex;

std::shared_ptr<ir::Module>
TensorBase::getHelperFunctions(const Format& format, Datatype ctype,
                               const std::vector<int>& dimensions) {
  helperFunctionsMutex.lock();
  const auto helperFunctionsReverse =
      util::ReverseConstIterable<TensorBase::HelperFuncsCache>(helperFunctions);
  for (const auto& helperFuncs : helperFunctionsReverse) {
    if (std::get<0>(helperFuncs) == format &&
        std::get<1>(helperFuncs) == ctype &&
        std::get<2>(helperFuncs) == dimensions) {
      // If helper functions had already been generated for specified tensor
      // format and type, then use cached version.
      const auto helperFuncsModule = std::get<3>(helperFuncs);
      helperFunctionsMutex.unlock();
      return helperFuncsModule;
    }
  }
  helperFunctionsMutex.unlock();

  std::shared_ptr<Module> helperModule = std::make_shared<Module>();

  std::function<Dimension(int)> getDim = [](int dim) {
    return Dimension(dim);
  };
  const auto dims = util::map(dimensions, getDim);

  if (format.getOrder() > 0) {
    const Format bufferFormat = COO(format.getOrder(), false, true, false,
                                    format.getModeOrdering());
    TensorVar bufferTensor(Type(ctype, Shape(dims)), bufferFormat);
    TensorVar packedTensor(Type(ctype, Shape(dims)), format);

    // Define packing and iterator routines in index notation.
    // TODO: Use `generatePackCOOStmt` function to generate pack routine.
    std::vector<IndexVar> indexVars(format.getOrder());
    IndexStmt packStmt = (packedTensor(indexVars) = bufferTensor(indexVars));
    IndexStmt iterateStmt = Yield(indexVars, packedTensor(indexVars));
    for (int i = format.getOrder() - 1; i >= 0; --i) {
      int mode = format.getModeOrdering()[i];
      packStmt = forall(indexVars[mode], packStmt);
      iterateStmt = forall(indexVars[mode], iterateStmt);
    }

    bool doAppend = true;
    for (int i = format.getOrder() - 1; i >= 0; --i) {
      const auto modeFormat = format.getModeFormats()[i];
      if (modeFormat.isBranchless() && i != 0) {
        const auto parentModeFormat = format.getModeFormats()[i - 1];
        if (parentModeFormat.isUnique() || !parentModeFormat.hasAppend()) {
          doAppend = false;
          break;
        }
      }
    }
    if (!doAppend) {
      packStmt = packStmt.assemble(packedTensor, AssembleStrategy::Insert);
    }

    // Lower packing and iterator code.
    helperModule->addFunction(lower(packStmt, "pack", true, true));
    helperModule->addFunction(lower(iterateStmt, "iterate", false, true));
  } else {
    const Format bufferFormat = COO(1, false, true, false);
    TensorVar bufferVector(Type(ctype, Shape({1})), bufferFormat);
    TensorVar packedScalar(Type(ctype, dims), format);

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

  helperFunctionsMutex.lock();
  helperFunctions.emplace_back(format, ctype, dimensions, helperModule);
  helperFunctionsMutex.unlock();

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

  // Fill values must be the same
  if (!equals(a.getFillValue(), b.getFillValue())) {
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
  size_t numCoordinates = tensor.content->coordinateBufferUsed / tensor.content->coordinateSize;
  for (size_t i = 0; i < numCoordinates; i++) {
    int* ptr = (int*)&tensor.content->coordinateBuffer->data()[i * tensor.content->coordinateSize];
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

ostream& operator<<(ostream& os, TensorBase& tensor) {
  tensor.syncValues();
  vector<string> dimensionStrings;
  for (int dimension : tensor.getDimensions()) {
    dimensionStrings.push_back(to_string(dimension));
  }
  os << tensor.getName() << " (" << util::join(dimensionStrings, "x") << ") "
     << tensor.getFormat() << ":" << std::endl;

  // Print coordinates
  size_t numCoordinates = tensor.content->coordinateBufferUsed / tensor.content->coordinateSize;
  for (size_t i = 0; i < numCoordinates; i++) {
    int* ptr = (int*)&tensor.content->coordinateBuffer->data()[i*tensor.content->coordinateSize];
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
