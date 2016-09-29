#include "tensor.h"

#include "packed_tensor.h"
#include "tree.h"

using namespace std;

namespace tac {

std::shared_ptr<PackedTensor>
pack(const std::vector<int>& dimensions, internal::ComponentType ctype,
     const Format& format, const void* coords, const void* values) {

  auto packedTensor = make_shared<PackedTensor>();

  match(format,
  function<void(const Values*,Matcher*)>([](const Values* l, Matcher* m) {
    
  })
  );

  return packedTensor;
}

}
