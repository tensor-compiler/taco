#include "taco/storage/index.h"

#include <iostream>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"

using namespace std;

namespace taco {
namespace storage {

// class Index
struct Index::Content {
  Format format;
  vector<ModeIndex> indices;
};

Index::Index() : content(new Content) {
}

Index::Index(const Format& format, const std::vector<ModeIndex>& indices)
    : Index() {
  taco_iassert(format.getOrder() == indices.size()  );
  content->format = format;
  content->indices = indices;
}

const Format& Index::getFormat() const {
  return content->format;
}

size_t Index::numModeIndices() const {
  return getFormat().getOrder();
}

const ModeIndex& Index::getModeIndex(int i) const {
  return content->indices[i];
}

ModeIndex Index::getModeIndex(int i) {
  taco_iassert(size_t(i) < getFormat().getOrder());
  return content->indices[i];
}

size_t Index::getSize() const {
  size_t size = 1;
  for (size_t i = 0; i < getFormat().getOrder(); i++) {
    auto dimType  = getFormat().getModeTypes()[i];
    auto dimIndex = getModeIndex(i);
    switch (dimType) {
      case ModeType::Dense:
        size *= getValue<size_t>(dimIndex.getIndexArray(0), 0);
        break;
      case ModeType::Sparse:
        size = getValue<size_t>(dimIndex.getIndexArray(0), size);
        break;
      case ModeType::Fixed:
        size *= getValue<size_t>(dimIndex.getIndexArray(0), 0);
        break;
    }
  }
  return size;
}

std::ostream& operator<<(std::ostream& os, const Index& index) {
  auto& format = index.getFormat();
  for (size_t i = 0; i < format.getOrder(); i++) {
    os << format.getModeTypes()[i] << " (" << format.getModeOrder()[i] << "): ";
    auto dimIndex = index.getModeIndex(i);
    for (size_t j = 0; j < dimIndex.numIndexArrays(); j++) {
      os << endl << "  " << dimIndex.getIndexArray(j);
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

size_t ModeIndex::numIndexArrays() const {
  return content->indexArrays.size();
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

}}
