#include "test.h"
#include "taco/tensor.h"

#include "taco/util/strings.h"

int main(int argc, char **argv) {
  // If there is just one argument and it is not a gtest option, then filter
  // the tests using that argument surrounded by wildcards.
  std::string filter;
  if (argc == 2 && std::string(argv[argc-1]).substr(0,2) != "--") {
    filter = std::string(argv[1]);

    char *dotPtr = strchr(argv[1], '.');
    if (!dotPtr) {
      filter = "*" + filter + "*";
      }
      else if (dotPtr[1] == '\0') {
        filter = filter + "*";
      }

      filter = std::string("--gtest_filter=") + filter;
      argv[1] = (char*)filter.c_str();
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#define STRINGIFY(x) #x

namespace taco {
namespace test {

std::string testDirectory() {
  return TO_STRING(TACO_TEST_DIR);
}

std::string testDataDirectory() {
  return testDirectory() + "/data/";
}

}}
