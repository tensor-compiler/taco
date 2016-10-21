#include "test.h"

#include "tensor.h"

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


namespace taco {
namespace test {

struct TestData::Content {
  Content(Tensor<double> tensor, TestData::Indices expectedIndices,
          vector<double> expectedValues)
      : tensor(tensor), expectedIndices(expectedIndices),
        expectedValues(expectedValues) {
  }

  Tensor<double> tensor;

  // Expected values
  TestData::Indices expectedIndices;
  vector<double> expectedValues;
};

// class TensorTestData
TestData::TestData(const Tensor<double>& tensor,
                               const Indices& expectedIndices,
                               const vector<double>& expectedValues)
    : content(new Content(tensor, expectedIndices, expectedValues)) {
}

const Tensor<double>& TestData::getTensor() const {
  return content->tensor;
}

const TestData::Indices& TestData::getExpectedIndices() const {
  return content->expectedIndices;
}

const vector<double>& TestData::getExpectedValues() const {
  return content->expectedValues;
}

ostream &operator<<(ostream& os, const TestData& data) {
  os << util::join(data.getTensor().getDimensions(), "x")
     << " (" << data.getTensor().getFormat() << ")";
  return os;
}

}}
