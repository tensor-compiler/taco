#include "tensor.h"

#include "packed_tensor.h"
#include "format.h"
#include "tree.h"

using namespace std;

namespace tac {

typedef PackedTensor::IndexType IndexType;

std::shared_ptr<PackedTensor>
pack(const std::vector<int>& dimensions, internal::ComponentType ctype,
     const Format& format, size_t ncoords, const void* coords,
     const void* values) {

  vector<vector<PackedTensor::IndexType*>> indices;

  // Compute the size of the values array
  size_t nnz = 1;
  auto& levels = format.getLevels();
  for (size_t i=0; i < levels.size(); ++i) {
    auto& level = levels[i];
    switch (level->type) {
      case Level::Dense: {
        std::cout << "Creating dense index (lvl " << i << ")" << std::endl;
        std::cout << std::endl;

        // A dense level increases the number of nnz
        nnz *= dimensions[i];
        break;
      }
      case Level::Sparse: {
        std::cout << "Creating sparse index (lvl " << i << ")" << std::endl;

        std::cout << "  segments: " << nnz+1 << std::endl;
        std::cout << "  indices:  " << ncoords << std::endl;
        std::cout << std::endl;

        // One segment per value on the previous level (nnz segments)
        auto segmentArray = (IndexType*)malloc((nnz+1) * sizeof(IndexType));

        // A sparse level packs nnz down to ncoords
        auto indexArray = (IndexType*)malloc(ncoords * sizeof(IndexType));
        nnz = ncoords;

        indices.push_back({segmentArray, indexArray});
        break;
      }
      case Level::Values: {
        break;  // Do nothing
      }
    }
  }
  void* vals = malloc(nnz * ctype.bytes());

  return make_shared<PackedTensor>(nnz, vals, indices);
}

}
