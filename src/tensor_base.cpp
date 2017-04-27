#include "taco/tensor_base.h"

#include <sstream>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/format.h"
#include "ir/ir.h"
#include "taco/storage/storage.h"
#include "taco/storage/pack.h"
#include "lower/lower.h"
#include "lower/iteration_schedule.h"
#include "backends/module.h"
#include "taco/io/hb_file_format.h"
#include "taco/io/mtx_file_format.h"
#include "taco/io/tns_file_format.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"

using namespace std;
using namespace taco::ir;
using namespace taco::storage;

namespace taco {

struct TensorBase::Content {
  string                   name;
  vector<int>              dimensions;
  ComponentType            ctype;

  storage::TensorStorage   storage;

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
    : TensorBase(name, ctype, {}, Format(), 1)  {
}

TensorBase::TensorBase(string name, ComponentType ctype, vector<int> dimensions)
  : TensorBase(name, ctype, dimensions,
               Format(vector<LevelType>(dimensions.size(),LevelType::Sparse))) {
}

TensorBase::TensorBase(ComponentType ctype, vector<int> dimensions)
    : TensorBase(util::uniqueName('A'), ctype, dimensions) {
}

TensorBase::TensorBase(ComponentType ctype, vector<int> dimensions,
                       Format format, size_t allocSize)
    : TensorBase(util::uniqueName('A'), ctype, dimensions, format, allocSize) {
}

TensorBase::TensorBase(string name, ComponentType ctype, vector<int> dimensions,
                       Format format, size_t allocSize) : content(new Content) {
  taco_uassert(format.getLevels().size() == dimensions.size())
      << "The number of format levels (" << format.getLevels().size()
      << ") must match the tensor order (" << dimensions.size() << ")";
  taco_uassert(ctype == ComponentType::Double)
      << "Only double tensors currently supported";

  content->name = name;
  content->dimensions = dimensions;
  content->storage = TensorStorage(format);
  content->ctype = ctype;
  content->allocSize = allocSize;

  // Initialize dense storage dimensions
  vector<Level> levels = format.getLevels();
  for (size_t i=0; i < levels.size(); ++i) {
    if (levels[i].getType() == LevelType::Dense) {
      TensorStorage::LevelIndex& index = content->storage.getLevelIndex(i);
      index.ptr = (int*)malloc(sizeof(int));
      index.ptr[0] = dimensions[i];
    }
  }
  
  content->module = make_shared<Module>();

  this->coordinateBuffer = shared_ptr<vector<char>>(new vector<char>);
  this->coordinateBufferUsed = 0;
  this->coordinateSize = getOrder()*sizeof(int) + ctype.bytes();
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

const ComponentType& TensorBase::getComponentType() const {
  return content->ctype;
}

const vector<taco::Var>& TensorBase::getIndexVars() const {
  return content->indexVars;
}

const taco::Expr& TensorBase::getExpr() const {
  return content->expr;
}

const storage::TensorStorage& TensorBase::getStorage() const {
  return content->storage;
}

storage::TensorStorage TensorBase::getStorage() {
  return content->storage;
}

size_t TensorBase::getAllocSize() const {
  return content->allocSize;
}

void TensorBase::setFormat(Format format) {
  content->storage.setFormat(format);
}

void TensorBase::setCSR(double* vals, int* rowPtr, int* colIdx) {
  taco_uassert(getFormat().isCSR()) <<
      "setCSR: the tensor " << getName() << " is not defined in the CSR format";
  auto S = getStorage();
  std::vector<int> denseDim = {getDimensions()[0]};
  TensorStorage::LevelIndex d0Index(util::copyToArray(denseDim), nullptr);
  TensorStorage::LevelIndex d1Index(rowPtr, colIdx);
  S.setLevelIndex(0, d0Index);
  S.setLevelIndex(1, d1Index);
  S.setValues(vals);
}

void TensorBase::getCSR(double** vals, int** rowPtr, int** colIdx) {
  taco_uassert(getFormat().isCSR()) <<
      "getCSR: the tensor " << getName() << " is not defined in the CSR format";
  auto S = getStorage();
  *vals = S.getValues();
  *rowPtr = S.getLevelIndex(1).ptr;
  *colIdx = S.getLevelIndex(1).idx;
}

void TensorBase::setCSC(double* vals, int* colPtr, int* rowIdx) {
  taco_uassert(getFormat().isCSC()) <<
      "setCSC: the tensor " << getName() << " is not defined in the CSC format";
  auto S = getStorage();
  std::vector<int> denseDim = {getDimensions()[1]};
  TensorStorage::LevelIndex d0Index(util::copyToArray(denseDim), nullptr);
  TensorStorage::LevelIndex d1Index(colPtr, rowIdx);
  S.setLevelIndex(0, d0Index);
  S.setLevelIndex(1, d1Index);
  S.setValues(vals);
}

void TensorBase::getCSC(double** vals, int** colPtr, int** rowIdx) {
  taco_uassert(getFormat().isCSC()) <<
      "getCSC: the tensor " << getName() << " is not defined in the CSC format";

  auto S = getStorage();
  *vals = S.getValues();
  *colPtr = S.getLevelIndex(1).ptr;
  *rowIdx = S.getLevelIndex(1).idx;
}

void TensorBase::read(std::string filename) {
  std::string extension = filename.substr(filename.find_last_of(".") + 1);
  if(extension == "rb") {
    readHB(filename);
  }
  else if (extension == "mtx") {
    if (getOrder()==2)
      readMTX(filename, 1);
    else
      readMTX(filename, getDimensions()[2]);
  }
  else if (extension == "tns") {
    readTNS(filename);
  }
  else {
    taco_uerror << "file extension not supported " << filename << std::endl;
  }
}

void TensorBase::readHB(std::string filename) {
  taco_uassert(getFormat().isCSC()) <<
      "readHB: the tensor " << getName() << " is not defined in the CSC format";
  std::ifstream hbfile;

  hbfile.open(filename.c_str());
  taco_uassert(hbfile.is_open()) <<
      " Error opening the file " << filename.c_str();
  int nrow, ncol;
  int *colptr = NULL;
  int *rowind = NULL;
  double *values = NULL;

  hb::readFile(hbfile, &nrow, &ncol, &colptr, &rowind, &values);
  taco_uassert((nrow==getDimensions()[0]) && (ncol==getDimensions()[1])) <<
      "readHB: the tensor " << getName() <<
      " does not have the same dimension in its declaration and HBFile" <<
  filename.c_str();

  auto storage = getStorage();
  std::vector<int> denseDim = {getDimensions()[1]};
  TensorStorage::LevelIndex d0Index(util::copyToArray(denseDim), nullptr);
  TensorStorage::LevelIndex d1Index(colptr, rowind);
  storage.setLevelIndex(0, d0Index);
  storage.setLevelIndex(1, d1Index);
  storage.setValues(values);

  hbfile.close();
}

void TensorBase::writeHB(std::string filename) const {
  taco_uassert(getFormat().isCSC()) <<
      "writeHB: the tensor " << getName() <<
      " is not defined in the CSC format";
  std::ofstream HBfile;

  HBfile.open(filename.c_str());
  taco_uassert(HBfile.is_open()) <<
      " Error opening the file " << filename.c_str();

  auto S = getStorage();
  auto size = S.getSize();

  double *values = S.getValues();
  int *colptr = S.getLevelIndex(1).ptr;
  int *rowind = S.getLevelIndex(1).idx;
  int nrow = getDimensions()[0];
  int ncol = getDimensions()[1];
  int nnzero = size.values;
  std::string key = getName();
  int valsize = size.values;
  int ptrsize = size.indexSizes[1].ptr;
  int indsize = size.indexSizes[1].idx;

  hb::writeFile(HBfile,const_cast<char*> (key.c_str()),
                nrow,ncol,nnzero,
                ptrsize,indsize,valsize,
                colptr,rowind,values);

  HBfile.close();
}

void TensorBase::readMTX(std::string filename, int blockSize) {
  std::ifstream MTXfile;

  MTXfile.open(filename.c_str());
  taco_uassert(MTXfile.is_open())
      << " Error opening the file " << filename.c_str() ;

  int nrow,ncol,nnzero;
  mtx::readFile(MTXfile, blockSize, &nrow, &ncol, &nnzero, this);
  if (blockSize == 1) {
    taco_uassert((nrow==getDimensions()[0])&&(ncol==getDimensions()[1])) <<
      "readMTX: the tensor " << getName() <<
      " does not have the same dimension in its declaration and MTXFile" <<
      filename.c_str();
  }
  else {
    taco_uassert((nrow/blockSize==getDimensions()[0])&&
                 (ncol/blockSize==getDimensions()[1])&&
                 (blockSize==getDimensions()[2])&&
                 (blockSize==getDimensions()[3])) <<
      "readMTX: the tensor " << getName() <<
      " does not have the same dimension in its declaration and MTXFile" <<
      filename.c_str();
  }
  MTXfile.close();
}
  
void TensorBase::writeMTX(std::string filename) const {
  std::ofstream MTXfile;

  MTXfile.open(filename.c_str());
  taco_uassert(MTXfile.is_open())
          << " Error opening the file " << filename.c_str();

  auto S = getStorage();
  auto size = S.getSize();

  int nnzero = size.values;
  std::string name = getName();

  mtx::writeFile(MTXfile, name, getDimensions() ,nnzero);

  for (const auto& val : *this) {
    for (size_t i=0; i<val.loc.size(); i++) {
      MTXfile << val.loc[i]+1 << " " ;
    }
    if (std::floor(val.dval) == val.dval)
      MTXfile << val.dval << ".0 " << std::endl;
    else
      MTXfile << val.dval << " " << std::endl;
  }

  MTXfile.close();
}

void TensorBase::readTNS(std::string filename) {
  std::ifstream TNSfile;

  TNSfile.open(filename.c_str());
  taco_uassert(TNSfile.is_open())
      << " Error opening the file " << filename.c_str();

  std::vector<int> dims;
  tns::readFile(TNSfile, dims, this);
  taco_uassert(dims == getDimensions()) << "readTNS: the tensor " << getName() 
      << " does not have the same dimension in its declaration and TNSFile" 
      << filename.c_str();

  TNSfile.close();
}

void TensorBase::writeTNS(std::string filename) const {
  std::ofstream TNSfile;

  TNSfile.open(filename.c_str());
  taco_uassert(TNSfile.is_open())
          << " Error opening the file " << filename.c_str();

  tns::writeFile(TNSfile, getName(), this);

  TNSfile.close();
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
  char* permutedCoordinates = coordinateBuffer->data();


  // The pack code expects the coordinates to be sorted
  numIntegersToCompare = order;
  qsort(permutedCoordinates, numCoordinates, coordSize, lexicographicalCmp);


  // Move coords into separate arrays
  std::vector<std::vector<int>> coordinates(order);
  for (size_t i=0; i < order; ++i) {
    coordinates[i] = std::vector<int>(numCoordinates);
  }

  std::vector<double> values(numCoordinates);
  for (size_t i=0; i < numCoordinates; ++i) {
    int* coordLoc = (int*)&permutedCoordinates[i*coordSize];
    for (size_t d=0; d < order; ++d) {
      coordinates[d][i] = *coordLoc;
      coordLoc++;
    }
    values[i] = *((double*)coordLoc);
  }
  taco_iassert(coordinates.size() > 0);

  // Pack indices and values
  content->storage = storage::pack(permutedDimensions, getFormat(),
                                   coordinates, values);
}

void TensorBase::zero() {
  auto resultStorage = getStorage();
  // Set values to 0.0 in case we are doing a += operation
  memset(resultStorage.getValues(), 0, content->valuesSize * sizeof(double));
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
    TensorStorage::LevelIndex levelIndex = resultStorage.getLevelIndex(i);
    auto& levelFormat = resultFormat.getLevels()[i];
    switch (levelFormat.getType()) {
      case Dense:
        arguments.push_back((void*)levelIndex.ptr);
        break;
      case Sparse:
      case Fixed:
        arguments.push_back((void*)levelIndex.ptr);
        arguments.push_back((void*)levelIndex.idx);
        break;
      case Offset:
      case Replicated:
        taco_not_supported_yet;
        break;
    }
  }
  arguments.push_back((void*)resultStorage.getValues());

  // Pack operand tensors
  vector<TensorBase> operands = expr_nodes::getOperands(tensor.getExpr());
  for (auto& operand : operands) {
    TensorStorage storage = operand.getStorage();
    Format format = storage.getFormat();
    for (size_t i=0; i<format.getLevels().size(); i++) {
      TensorStorage::LevelIndex levelIndex = storage.getLevelIndex(i);
      auto& levelFormat = format.getLevels()[i];
      switch (levelFormat.getType()) {
        case Dense:
          arguments.push_back((void*)levelIndex.ptr);
          break;
        case Sparse:
        case Fixed:
          arguments.push_back((void*)levelIndex.ptr);
          arguments.push_back((void*)levelIndex.idx);
          break;
        case Offset:
        case Replicated:
          taco_not_supported_yet;
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

void TensorBase::setExpr(taco::Expr expr) {
  content->expr = expr;

  storage::TensorStorage storage = getStorage();
  Format format = storage.getFormat();
  auto& levels = format.getLevels();
  for (size_t i=0; i < levels.size(); ++i) {
    Level level = levels[i];
    auto& levelIndex = storage.getLevelIndex(i);
    switch (level.getType()) {
      case LevelType::Dense:
        break;
      case LevelType::Sparse:
        levelIndex.ptr = (int*)malloc(getAllocSize() * sizeof(int));
        levelIndex.ptr[0] = 0;
        levelIndex.idx = (int*)malloc(getAllocSize() * sizeof(int));
        break;
      case LevelType::Fixed:
        levelIndex.ptr = (int*)malloc(sizeof(int));
        levelIndex.idx = (int*)malloc(getAllocSize() * sizeof(int));
        break;
      case LevelType::Offset:
      case LevelType::Replicated:
        taco_not_supported_yet;
        break;
    }
  }
}

void TensorBase::setIndexVars(vector<taco::Var> indexVars) {
  content->indexVars = indexVars;
}

void TensorBase::printComputeIR(std::ostream& os, bool color) const {
  IRPrinter printer(os,color);
  content->computeFunc.as<Function>()->body.accept(&printer);
}

void TensorBase::printAssemblyIR(std::ostream& os, bool color) const {
  IRPrinter printer(os,color);
  content->assembleFunc.as<Function>()->body.accept(&printer);
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
  auto ait = a.begin();
  auto bit = b.begin();

  for (; ait != a.end() && bit != b.end(); ++ait, ++bit) {
    if (ait->loc != bit->loc) {
      return false;
    }
    if (abs((ait->dval-bit->dval)/ait->dval) > 10e-6) {
      return false;
    }
  }

  return (ait == a.end() && bit == b.end());
}

bool operator==(const TensorBase& l, const TensorBase& r) {
  return l.content == r.content;
}

bool operator<(const TensorBase& l, const TensorBase& r) {
  return l.content < r.content;
}

void TensorBase::assembleInternal() {
  content->module->callFunc("assemble", content->arguments.data());
  
  size_t j = 0;
  auto resultStorage = getStorage();
  auto resultFormat = resultStorage.getFormat();
  for (size_t i=0; i<resultFormat.getLevels().size(); i++) {
    TensorStorage::LevelIndex& levelIndex = resultStorage.getLevelIndex(i);
    auto& levelFormat = resultFormat.getLevels()[i];
    switch (levelFormat.getType()) {
      case Dense:
        j++;
        break;
      case Sparse:
      case Fixed:
        levelIndex.ptr = (int*)content->arguments[j++];
        levelIndex.idx = (int*)content->arguments[j++];
        break;
      case Offset:
      case Replicated:
        taco_not_supported_yet;
        break;
    }
  }

  content->valuesSize = resultStorage.getSize().values;
  resultStorage.setValues((double*)malloc(content->valuesSize*sizeof(double)));
  content->arguments[j] = resultStorage.getValues();
}

void TensorBase::computeInternal() {
  this->content->module->callFunc("compute", content->arguments.data());
}

ostream& operator<<(ostream& os, const TensorBase& t) {
  vector<string> dimStrings;
  for (int dim : t.getDimensions()) {
    dimStrings.push_back(to_string(dim));
  }
  os << t.getName()
     << " (" << util::join(dimStrings, "x") << ", " << t.getFormat() << ")";

  if (t.getStorage().defined()) {
    // Print packed data
    os << endl << t.getStorage();
  }

  return os;
}

TensorBase readTensor(std::string filename, std::string name) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;

  if (name=="") {
    name = filename.substr(filename.find_last_of("/") + 1);
    name = filename.substr(name.find_first_of(".") + 1);
  }

  string extension = filename.substr(filename.find_last_of(".") + 1);
  TensorFileFormat fileFormat;
  if (extension == "mtx") {
    fileFormat = TensorFileFormat::mtx;
  }
  else if (extension == "tns") {
    fileFormat = TensorFileFormat::tns;
  }
  else {
    fileFormat = TensorFileFormat::tns;  // suppress warning
    taco_uerror << "File extension not recognized: " << filename << std::endl;
  }

  TensorBase tensor = readTensor(file, fileFormat, name);
  file.close();
  return tensor;
}

TensorBase readTensor(istream& stream, TensorFileFormat fileFormat,
                      string name) {
  TensorBase tensor;
  switch (fileFormat) {
    case TensorFileFormat::mtx:
      tensor = mtx::readTensor(stream, name);
      break;
    case TensorFileFormat::tns:
      tensor = tns::readTensor(stream, name);
      break;
    case TensorFileFormat::hb:
      taco_not_supported_yet;
      break;
  }
  return tensor;
}

void writeTensor(string filename, const TensorBase& tensor) {
}


void writeTensor(ofstream& file, const TensorBase& tensor,
                 TensorFileFormat fileFormat) {
}

}
