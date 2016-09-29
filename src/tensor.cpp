#include "tensor.h"

#include "packed_tensor.h"
#include "format.h"
#include "tree.h"

using namespace std;

namespace tac {

std::shared_ptr<PackedTensor>
pack(const std::vector<int>& dimensions, internal::ComponentType ctype,
     const Format& format, size_t ncoords, const void* coords,
     const void* values) {

  // Compute the size of the values array
  size_t nnz = 1;
  auto& levels = format.getLevels();
  for (size_t i=0; i < levels.size(); ++i) {
    auto& level = levels[i];
    switch (level->type) {
      case Level::Dense: {
        nnz *= dimensions[i];
        break;
      }
      case Level::Sparse: {
        nnz = ncoords;
        break;
      }
      case Level::Values:
        break;  // Do nothing
    }
  }
  std::cout << nnz << std::endl;
  void* vals = const_cast<void*>(malloc(nnz * ctype.bytes()));

  vector<vector<uint32_t*>> indices;

  return make_shared<PackedTensor>(ncoords, vals, indices);
}

}
