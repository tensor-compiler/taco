#include "test.h"
#include "taco/tensor.h"

#include "taco/util/strings.h"
#include "taco/storage/storage.h"

int main(int argc, char **argv) {
  // If there is just one argument and it is not a gtest option, then filter
  // the tests using that argument surrounded by wildcards.
  std::string filter;
  if (argc == 2 && std::string(argv[argc-1]).substr(0,2) != "--") {
    filter = std::string(argv[1]);
    filter = "*" + filter + "*";
    filter = std::string("--gtest_filter=") + filter;
    argv[1] = (char*)filter.c_str();
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#define STRINGIFY(x) #x

namespace taco {
namespace test {

void ASSERT_STORAGE_EQ(TensorStorage expected, TensorStorage actual) {
  SCOPED_TRACE("\nexpected:\n" + util::toString(expected) +
               "\nactual:\n" + util::toString(actual));
  ASSERT_TRUE(equals(expected, actual));
}

void ASSERT_TENSOR_EQ(TensorBase expected, TensorBase actual) {
  SCOPED_TRACE(string("expected: ") + util::toString(expected));
  SCOPED_TRACE(string("  actual: ") + util::toString(actual));
  ASSERT_TRUE(equals(expected, actual));
}

std::string testDirectory() {
  return TO_STRING(TACO_TEST_DIR);
}

std::string testDataDirectory() {
  return testDirectory() + "/data/";
}

ostream& operator<<(ostream& os, const NotationTest& test) {
  os << endl;
  os << "Expected: " << test.expected << endl;
  os << "Actual:   " << test.actual << endl;
  return os;
}

}}
