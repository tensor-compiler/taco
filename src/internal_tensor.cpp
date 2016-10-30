#include "internal_tensor.h"

#include <sstream>

#include "var.h"
#include "internal_tensor.h"
#include "storage/storage.h"
#include "format.h"
#include "ir.h"
#include "lower/lower.h"
#include "lower/iteration_schedule.h"
#include "backend_c.h"
#include "util/strings.h"

using namespace std;
using namespace taco::ir;
using namespace taco::storage;

namespace taco {
namespace internal {

// These are defined here to separate out the code here
// from the actual storage in PackedTensor
typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

struct Tensor::Content {
  string                   name;
  vector<int>              dimensions;
  ComponentType            ctype;

  std::vector<Coordinate>  coordinates;

  Format                   format;
  storage::Storage         storage;

  vector<taco::Var>        indexVars;
  taco::Expr               expr;
  vector<void*>            arguments;

  lower::IterationSchedule schedule;
  Stmt                     evaluateFunc;
  Stmt                     assembleFunc;
  shared_ptr<Module>       module;
};

Tensor::Tensor(string name, vector<int> dimensions, 
               Format format, ComponentType ctype)
    : content(new Content) {
  content->name = name;
  content->dimensions = dimensions;
  content->storage = Storage(format);
  content->ctype = ctype;
}

string Tensor::getName() const {
  return content->name;
}

size_t Tensor::getOrder() const {
  return content->dimensions.size();
}

const vector<int>& Tensor::getDimensions() const {
  return content->dimensions;
}

const Format& Tensor::getFormat() const {
  return content->storage.getFormat();
}

const ComponentType& Tensor::getComponentType() const {
  return content->ctype;
}

const vector<taco::Var>& Tensor::getIndexVars() const {
  return content->indexVars;
}

const taco::Expr& Tensor::getExpr() const {
  return content->expr;
}

const storage::Storage& Tensor::getStorage() const {
  return content->storage;
}

/// Count unique entries between iterators (assumes values are sorted)
static vector<int> getUniqueEntries(const vector<int>::const_iterator& begin,
                                    const vector<int>::const_iterator& end) {
  vector<int> uniqueEntries;
  if (begin != end) {
    size_t curr = *begin;
    uniqueEntries.push_back(curr);
    for (auto it = begin+1; it != end; ++it) {
      size_t next = *it;
      iassert(next >= curr);
      if (curr < next) {
        curr = next;
        uniqueEntries.push_back(curr);
      }
    }
  }
  return uniqueEntries;
}

static void packTensor(const vector<int>& dims,
                       const vector<vector<int>>& coords,
                       const double* vals,
                       size_t begin, size_t end,
                       const vector<Level>& levels, size_t i,
                       Indices* indices,
                       vector<double>* values) {

  // Base case: no more tree levels so we pack values
  if (i == levels.size()) {
    iassert(begin == end || begin == end-1);
    if (begin < end) {
      values->push_back(vals[begin]);
    }
    else {
      values->push_back(0.0);
    }
    return;
  }

  auto& level       = levels[i];
  auto& levelCoords = coords[i];
  auto& index       = (*indices)[i];

  switch (level.getType()) {
    case Dense: {
      // Iterate over each index value and recursively pack it's segment
      size_t cbegin = begin;
      for (int j=0; j < (int)dims[i]; ++j) {
        // Scan to find segment range of children
        size_t cend = cbegin;
        while (cend < end && levelCoords[cend] == j) {
          cend++;
        }
        packTensor(dims, coords, vals, cbegin, cend, levels, i+1,
                   indices, values);
        cbegin = cend;
      }
      break;
    }
    case Sparse: {
      auto indexValues = getUniqueEntries(levelCoords.begin()+begin,
                                          levelCoords.begin()+end);

      // Store segment end: the size of the stored segment is the number of
      // unique values in the coordinate list
      index[0].push_back(index[1].size() + indexValues.size());

      // Store unique index values for this segment
      index[1].insert(index[1].end(), indexValues.begin(), indexValues.end());

      // Iterate over each index value and recursively pack it's segment
      size_t cbegin = begin;
      for (int j : indexValues) {
        // Scan to find segment range of children
        size_t cend = cbegin;
        while (cend < end && levelCoords[cend] == j) {
          cend++;
        }
        packTensor(dims, coords, vals, cbegin, cend, levels, i+1,
                   indices, values);
        cbegin = cend;
      }
      break;
    }
    case Fixed: {
      not_supported_yet;
      break;
    }
  }
}

void Tensor::insert(const std::vector<int>& coord, int val) {
  iassert(getComponentType() == ComponentType::Int);
  content->coordinates.push_back(Coordinate(coord, val));
}

void Tensor::insert(const std::vector<int>& coord, float val) {
  iassert(getComponentType() == ComponentType::Float);
  content->coordinates.push_back(Coordinate(coord, val));
}

void Tensor::insert(const std::vector<int>& coord, double val) {
  iassert(getComponentType() == ComponentType::Double);
  content->coordinates.push_back(Coordinate(coord, val));
}

void Tensor::insert(const std::vector<int>& coord, bool val) {
  iassert(getComponentType() == ComponentType::Bool);
  content->coordinates.push_back(Coordinate(coord, val));
}

/// Pack the coordinates (stored as structure-of-arrays) according to the
/// tensor's format.
void Tensor::pack() {
  const std::vector<Level>& levels     = getFormat().getLevels();
  const std::vector<int>&   dimensions = getDimensions();

  iassert(levels.size() == getOrder());

  // Packing code currently only packs coordinates in the order of the
  // dimensions. To work around this we just permute each coordinate according
  // to the storage dimensions.
  std::vector<int> permutation;
  for (auto& level : levels) {
    permutation.push_back(level.getDimension());
  }

  std::vector<Coordinate> permutedCoords;
  permutation.reserve(content->coordinates.size());
  for (size_t i=0; i < content->coordinates.size(); ++i) {
    auto& coord = content->coordinates[i];
    std::vector<int> ploc(coord.loc.size());
    for (size_t j=0; j < getOrder(); ++j) {
      ploc[permutation[j]] = coord.loc[j];
    }

    switch (getComponentType().getKind()) {
      case ComponentType::Bool:
        permutedCoords.push_back(Coordinate(ploc, coord.bval));
        break;
      case ComponentType::Int:
        permutedCoords.push_back(Coordinate(ploc, coord.ival));
        break;
      case ComponentType::Float:
        permutedCoords.push_back(Coordinate(ploc, coord.fval));
        break;
      case ComponentType::Double:
        permutedCoords.push_back(Coordinate(ploc, coord.dval));
        break;
      default:
        not_supported_yet;
        break;
    }
  }
  content->coordinates.clear();

  // The pack code requires the coordinates to be sorted
  std::sort(permutedCoords.begin(), permutedCoords.end());

  // convert coords to structure of arrays
  std::vector<std::vector<int>> coords(getOrder());
  for (size_t i=0; i < getOrder(); ++i) {
    coords[i] = std::vector<int>(permutedCoords.size());
  }

  // FIXME: element type should not be hard-coded to double
  std::vector<double> vals(permutedCoords.size());

  for (size_t i=0; i < permutedCoords.size(); ++i) {
    for (size_t d=0; d < getOrder(); ++d) {
      coords[d][i] = permutedCoords[i].loc[d];
    }
    switch (getComponentType().getKind()) {
      case ComponentType::Bool:
        vals[i] = permutedCoords[i].bval;
        break;
      case ComponentType::Int:
        vals[i] = permutedCoords[i].ival;
        break;
      case ComponentType::Float:
        vals[i] = permutedCoords[i].fval;
        break;
      case ComponentType::Double:
        vals[i] = permutedCoords[i].dval;
        break;
      default:
        not_supported_yet;
        break;
    }
  }

  iassert(coords.size() > 0);
  size_t numCoords = coords[0].size();

  Indices indices;
  indices.reserve(levels.size()-1);

  // Create the vectors to store pointers to indices/index sizes
  size_t nnz = 1;
  for (size_t i=0; i < levels.size(); ++i) {
    auto& level = levels[i];
    switch (level.getType()) {
      case Dense: {
        indices.push_back({});
        nnz *= dimensions[i];
        break;
      }
      case Sparse: {
        // A sparse level packs nnz down to #coords
        nnz = numCoords;

        // Sparse indices have two arrays: a segment array and an index array
        indices.push_back({{}, {}});

        // Add start of first segment
        indices[i][0].push_back(0);
        break;
      }
      case Fixed: {
        not_supported_yet;
        break;
      }
    }
  }

  tassert(getComponentType() == ComponentType::Double)
      << "make the packing machinery work with other primitive types later. "
      << "Right now we're specializing to doubles so that we can use a "
      << "resizable vector, but eventually we should use a two pass pack "
      << "algorithm that figures out sizes first, and then packs the data";

  std::vector<double> values;

  // Pack indices and values
  packTensor(dimensions, coords, (const double*)vals.data(), 0, numCoords,
             levels, 0, &indices, &values);

  // Copy packed data into tensor storage
  for (size_t i=0; i < levels.size(); ++i) {
    LevelType levelType = levels[i].getType();

    int* ptr = nullptr;
    int* idx = nullptr;
    switch (levelType) {
      case LevelType::Dense:
        ptr = util::copyToArray({dimensions[i]});
        idx = nullptr;
        break;
      case LevelType::Sparse:
        ptr = util::copyToArray(indices[i][0]);
        idx = util::copyToArray(indices[i][1]);
        break;
      case LevelType::Fixed: {
        not_supported_yet;
        break;
      }
    }
    content->storage.setLevelIndex(i, ptr, idx);
  }
  content->storage.setValues(util::copyToArray(values));
}

void Tensor::compile() {
  iassert(getExpr().defined()) << "No expression defined for tensor";

  content->assembleFunc = lower::lower(*this, {lower::Assemble}, "assemble");
  content->evaluateFunc = lower::lower(*this, {lower::Evaluate}, "evaluate");

  stringstream cCode;
  CodeGen_C cg(cCode);
  cg.compile(content->assembleFunc);
  cg.compile(content->evaluateFunc);
  content->module = make_shared<Module>(cCode.str());
  content->module->compile();
}

void Tensor::assemble() {
  content->module->call_func("assemble", content->arguments.data());
}

void Tensor::evaluate() {
  content->module->call_func("evaluate", content->arguments.data());
}

static inline vector<void*> packArguments(const Tensor& tensor) {
  vector<void*> arguments;

  // Pack the result tensor
  auto resultStorage = tensor.getStorage();
  auto resultFormat = resultStorage.getFormat();
  for (size_t i=0; i<resultFormat.getLevels().size(); i++) {
    Storage::LevelIndex levelIndex = resultStorage.getLevelIndex(i);
    auto& levelFormat = resultFormat.getLevels()[i];
    switch (levelFormat.getType()) {
      case Dense:
        arguments.push_back((void*)levelIndex.ptr);
        break;
      case Sparse:
        arguments.push_back((void*)levelIndex.ptr);
        arguments.push_back((void*)levelIndex.idx);
        break;
      case Fixed:
        not_supported_yet;
        break;
    }
  }
  arguments.push_back((void*)resultStorage.getValues());

  // Pack operand tensors
  vector<Tensor> operands = getOperands(tensor.getExpr());
  for (auto& operand : operands) {
    Storage storage = operand.getStorage();
    Format format = storage.getFormat();
    for (size_t i=0; i<format.getLevels().size(); i++) {
      Storage::LevelIndex levelIndex = storage.getLevelIndex(i);
      auto& levelFormat = format.getLevels()[i];
      switch (levelFormat.getType()) {
        case Dense:
          arguments.push_back((void*)levelIndex.ptr);
          break;
        case Sparse:
          arguments.push_back((void*)levelIndex.ptr);
          arguments.push_back((void*)levelIndex.idx);
          break;
        case Fixed:
          not_supported_yet;
          break;
      }
    }
    arguments.push_back((void*)storage.getValues());
  }

  return arguments;
}

void Tensor::setExpr(taco::Expr expr) {
  content->expr = expr;
  content->arguments = packArguments(*this);
}

void Tensor::setIndexVars(vector<taco::Var> indexVars) {
  content->indexVars = indexVars;
}

void Tensor::printIterationSpace() const {
  string funcName = "print";
  // TODO: Remove lower::Assemble below
  auto print = lower::lower(*this, {lower::Print, lower::Assemble}, funcName);
  std::cout << std::endl << "# IR:" << std::endl;
  std::cout << print << std::endl;

  stringstream cCode;
  CodeGen_C cg(cCode);
  cg.compile(print);
  content->module = make_shared<Module>(cCode.str());
  content->module->compile();

//  std::cout << std::endl << "# Code" << std::endl << cCode.str();
  std::cout << std::endl << "# Output:" << std::endl;
  content->module->call_func(funcName, content->arguments.data());
}

bool operator!=(const Tensor& l, const Tensor& r) {
  return l.content != r.content;
}

bool operator<(const Tensor& l, const Tensor& r) {
  return l.content < r.content;
}

ostream& operator<<(ostream& os, const internal::Tensor& t) {
  vector<string> dimStrings;
  for (int dim : t.getDimensions()) {
    dimStrings.push_back(to_string(dim));
  }
  os << t.getName()
     << " (" << util::join(dimStrings, "x") << ", " << t.getFormat() << ")";

  // Print packed data
  if (t.getStorage().defined()) {
    os << endl << t.getStorage();
  }
    
  if (t.content->coordinates.size() > 0) {
    os << std::endl << "Coordinates: ";
    for (auto& coord : t.content->coordinates) {
      os << std::endl << "  (" << util::join(coord.loc) << "): ";
      switch (t.getComponentType().getKind()) {
        case ComponentType::Bool:
          os << coord.bval;
          break;
        case ComponentType::Int:
          os << coord.ival;
          break;
        case ComponentType::Float:
          os << coord.fval;
          break;
        case ComponentType::Double:
          os << coord.dval;
          break;
        default:
          not_supported_yet;
          break;
      }
    }
  }

  return os;
}

}}
