#include "taco/storage/index.h"

#include <iostream>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/util/array.h"

using namespace std;

namespace taco {
namespace storage {

// class Index
struct Index::Content {
  Format format;
  vector<DimensionIndex> indices;
};

Index::Index() : content(nullptr) {
}

Index::Index(const Format& format, const std::vector<DimensionIndex>& indices) :
    content(new Content) {
  content->format = format;
  content->indices = indices;
}

const Format& Index::getFormat() const {
  return content->format;
}

const DimensionIndex& Index::getDimensionIndex(int i) const {
  return content->indices[i];
}

std::ostream& operator<<(std::ostream& os, const Index& index) {
  auto& format = index.getFormat();
  for (size_t i = 0; i < format.getOrder(); i++) {
    os << format.getDimensionTypes()[i] << " ("
       << format.getDimensionOrder()[i] << "): ";
    auto dimIndex = index.getDimensionIndex(i);
    for (size_t j = 0; j < dimIndex.numIndexArrays(); j++) {
      os << endl << "  " << dimIndex.getIndexArray(j);
    }
    if (i < format.getOrder()-1) os << endl;
  }
  return os;
}


// class DimensionIndex
struct DimensionIndex::Content {
  vector<Array> indexArrays;
};

DimensionIndex::DimensionIndex(const std::vector<Array>& indexArrays)
    : content(new Content) {
  content->indexArrays = indexArrays;
}

size_t DimensionIndex::numIndexArrays() const {
  return content->indexArrays.size();
}

const Array& DimensionIndex::getIndexArray(int i) const {
  return content->indexArrays[i];
}

// Factory functions
Index makeCSRIndex(size_t numrows, int* rowptr, int* colidx) {
  return Index(CSR, {DimensionIndex({Array({(int)numrows})}),
                     DimensionIndex({Array(numrows+1, rowptr),
                                     Array(rowptr[numrows], colidx)})});
}

Index makeCSRIndex(const vector<int>& rowptr, const vector<int>& colidx) {
  return Index(CSR, {DimensionIndex({Array({(int)rowptr.size()})}),
                     DimensionIndex({Array(rowptr), Array(colidx)})});
}

}}
