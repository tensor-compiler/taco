#include "internal_tensor.h"

#include "internal_tensor.h"
#include "packed_tensor.h"
#include "format.h"
#include "tree.h"
#include "iteration_schedule.h"
#include "lower.h"

using namespace std;

namespace taco {

namespace internal {

typedef PackedTensor::IndexType  IndexType;
typedef PackedTensor::IndexArray IndexArray;
typedef PackedTensor::Index      Index;
typedef PackedTensor::Indices    Indices;

struct Tensor::Content {
  std::string                     name;
  std::vector<size_t>             dimensions;
  Format                          format;

  std::shared_ptr<PackedTensor>   packedTensor;

  std::vector<taco::Var>          indexVars;
  taco::Expr                      expr;

  IterationSchedule               schedule;
  std::shared_ptr<Stmt> code;
};

Tensor::Tensor(string name, vector<size_t> dimensions, Format format)
    : content(new Content) {
  content->name = name;
  content->dimensions = dimensions;
  content->format = format;
}

std::string Tensor::getName() const {
  return content->name;
}

size_t Tensor::getOrder() const {
  return content->dimensions.size();
}

const std::vector<size_t>& Tensor::getDimensions() const {
  return content->dimensions;
}

const Format& Tensor::getFormat() const {
  return content->format;
}

const std::vector<taco::Var>& Tensor::getIndexVars() const {
  return content->indexVars;
}

const taco::Expr& Tensor::getExpr() const {
  return content->expr;
}

const std::shared_ptr<PackedTensor> Tensor::getPackedTensor() const {
  return content->packedTensor;
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

static void packTensor(const vector<size_t>& dims,
                       const vector<vector<int>>& coords,
                       const double* vals,
                       size_t begin, size_t end,
                       const vector<Level>& levels, size_t i,
                       Indices* indices,
                       std::vector<double>* values) {

  auto& level       = levels[i];
  auto& levelCoords = coords[i];
  auto& index       = (*indices)[i];

  switch (level.type) {
    case Level::Dense: {
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
    case Level::Sparse: {
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
    case Level::Values: {
      iassert(begin == end || begin == end-1);
      if (begin < end) {
        values->push_back(vals[begin]);
      }
      else {
        values->push_back(0.0);
      }
      break;
    }
  }
}

void Tensor::pack(const std::vector<std::vector<int>>& coords,
                  ComponentType ctype, const void* vals) {
  iassert(coords.size() > 0);
  size_t numCoords = coords[0].size();

  const vector<Level>&  levels     = getFormat().getLevels();
  const vector<size_t>& dimensions = getDimensions();

  Indices indices;
  indices.reserve(levels.size()-1);

  // Create the vectors to store pointers to indices/index sizes
  size_t nnz = 1;
  for (size_t i=0; i < levels.size(); ++i) {
    auto& level = levels[i];
    switch (level.type) {
      case Level::Dense: {
        indices.push_back({});
        nnz *= dimensions[i];
        break;
      }
      case Level::Sparse: {
        // A sparse level packs nnz down to #coords
        nnz = numCoords;

        // Sparse indices have two arrays: a segment array and an index array
        indices.push_back({{}, {}});

        // Add start of first segment
        indices[i][0].push_back(0);
        break;
      }
      case Level::Values: {
        // Do nothing
        break;
      }
    }
  }

  tassert(ctype == ComponentType::Double)
      << "make the packing machinery work with other primitive types later. "
      << "Right now we're specializing to doubles so that we can use a "
      << "resizable std::vector, but eventually we should use a two pass pack "
      << "algorithm that figures out sizes first, and then packs the data";

  std::vector<double> values;

  // Pack indices and values
  packTensor(dimensions, coords, (const double*)vals, 0, numCoords,
             levels, 0, &indices, &values);

  content->packedTensor = make_shared<PackedTensor>(values, indices);
}

void Tensor::compile() {
  iassert(getExpr().defined()) << "No expression defined for tensor";
//  content->code = lower(*this);
}

void Tensor::assemble() {
}

void Tensor::evaluate() {
}


void Tensor::setExpr(taco::Expr expr) {
  content->expr = expr;
}

void Tensor::setIndexVars(std::vector<taco::Var> indexVars) {
  content->indexVars = indexVars;
}

std::ostream& operator<<(std::ostream& os, const internal::Tensor& t) {
  std::vector<std::string> dimStrings;
  for (int dim : t.getDimensions()) {
    dimStrings.push_back(std::to_string(dim));
  }
  os << t.getName()
  << " (" << util::join(dimStrings, "x") << ", " << t.getFormat() << ")";

  // Print packed data
  if (t.getPackedTensor() != nullptr) {
    os << std::endl << *t.getPackedTensor();
  }
  return os;
}

}}
