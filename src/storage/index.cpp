#include "taco/storage/index.h"

#include <iostream>
#include <vector>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/array.h"

using namespace std;

namespace taco {

// class Index
struct Index::Content {
  Format format;
  vector<ModeIndex> indices;
};

Index::Index() : content(new Content) {
}

Index::Index(const Format& format) : Index() {
  content->format = format;
  content->indices = vector<ModeIndex>(format.getOrder());
}

Index::Index(const Format& format, const std::vector<ModeIndex>& indices)
    : Index() {
  taco_iassert((size_t)format.getOrder() == indices.size());
  content->format = format;
  content->indices = indices;
}

const Format& Index::getFormat() const {
  return content->format;
}

int Index::numModeIndices() const {
  return getFormat().getOrder();
}

const ModeIndex& Index::getModeIndex(int i) const {
  return content->indices[i];
}

ModeIndex Index::getModeIndex(int i) {
  taco_iassert(i < getFormat().getOrder())
      << "mode: " << i << endl << "order: " << getFormat().getOrder();
  return content->indices[i];
}

size_t Index::getSize() const {
  size_t size = 1;
  for (int i = 0; i < getFormat().getOrder(); i++) {
    auto modeType  = getFormat().getModeFormats()[i];
    auto modeIndex = getModeIndex(i);
    if (modeType.getName() == Dense.getName()) {
      size *= modeIndex.getIndexArray(0).get(0).getAsIndex();
    } else if (modeType.getName() == Sparse.getName()) {
      size = modeIndex.getIndexArray(0).get(size).getAsIndex();
    } else {
      taco_not_supported_yet;
    }
  }
  return size;
}

std::ostream& operator<<(std::ostream& os, const Index& index) {
  auto& format = index.getFormat();
  for (int i = 0; i < format.getOrder(); i++) {
    os << format.getModeFormats()[i] <<
      " (" << format.getModeOrdering()[i] << "): ";
    auto modeIndex = index.getModeIndex(i);
    for (int j = 0; j < modeIndex.numIndexArrays(); j++) {
      os << endl << "  " << modeIndex.getIndexArray(j);
    }
    if (i < format.getOrder()-1) os << endl;
  }
  return os;
}


// class ModeIndex
struct ModeIndex::Content {
  vector<Array> indexArrays;
};

ModeIndex::ModeIndex() : content(new Content) {
}

ModeIndex::ModeIndex(const std::vector<Array>& indexArrays) : ModeIndex() {
  content->indexArrays = indexArrays;
}

int ModeIndex::numIndexArrays() const {
  return (int)content->indexArrays.size();
}

const Array& ModeIndex::getIndexArray(int i) const {
  return content->indexArrays[i];
}

Array ModeIndex::getIndexArray(int i) {
  return content->indexArrays[i];
}

// Factory functions
Index makeCSRIndex(size_t numrows, int* rowptr, int* colidx) {
  return Index(CSR, {ModeIndex({makeArray({(int)numrows})}),
                     ModeIndex({makeArray(rowptr, numrows+1),
                                makeArray(colidx, rowptr[numrows])})});
}

Index makeCSRIndex(const vector<int>& rowptr, const vector<int>& colidx) {
  return Index(CSR, {ModeIndex({makeArray({(int)(rowptr.size()-1)})}),
                     ModeIndex({makeArray(rowptr), makeArray(colidx)})});
}

Index makeCSCIndex(size_t numcols, int* colptr, int* rowidx) {
  return Index(CSC, {ModeIndex({makeArray({(int)numcols})}),
                     ModeIndex({makeArray(colptr, numcols+1),
                                makeArray(rowidx, colptr[numcols])})});
}

Index makeCSCIndex(const vector<int>& colptr, const vector<int>& rowidx) {
  return Index(CSC, {ModeIndex({makeArray({(int)(colptr.size()-1)})}),
                     ModeIndex({makeArray(colptr), makeArray(rowidx)})});
}

}
