#include "test.h"
#include "test_tensors.h"
#include "expr_factory.h"

#include <fstream>

#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/util/env.h"

#include <cmath>
#include <stdio.h>

using namespace taco;

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Idx;        // [0,2] index arrays per Index
typedef std::vector<Idx>        Indices;    // One Index per level


class APIMatrixStorageTestData {
public:
	APIMatrixStorageTestData(string tensorFile, const Indices& expectedIndices,
                            const vector<double> expectedValues)
      : tensor(readTestTensor(tensorFile, CSC)),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
    tensor.pack();
  }

	APIMatrixStorageTestData(TensorBase tensor, const Indices& expectedIndices,
                            const vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }

  TensorBase getTensor() const {
    return tensor;
  }

  const Indices& getExpectedIndices() const {
    return expectedIndices;
  }

  const vector<double>& getExpectedValues() const {
    return expectedValues;
  }

private:
  TensorBase     tensor;
  Indices        expectedIndices;
  vector<double> expectedValues;
};

class APIFileTestData {
public:
  APIFileTestData(std::string filename) : initTensor(true), filename(filename) {
  }

  APIFileTestData(TensorBase tensor, std::string filename)
      : tensor(tensor), filename(filename) {
  }

  APIFileTestData(std::string filename, Format format)
      : filename(filename) {
    std::string mtxfile = filename + ".mtx";
    tensor = readTestTensor(mtxfile, format);
  }

  TensorBase getTensor(Format format) const {
    return initTensor ? readTestTensor(filename, format) : tensor;
  }

  string getFilename() const {
    return filename;
  }

private:
  bool initTensor = false;
  TensorBase tensor;
  std::string filename;
};

struct apiset : public TestWithParam<APIMatrixStorageTestData> {};
struct apiget : public TestWithParam<APIMatrixStorageTestData> {};
struct apiwrb : public TestWithParam<APIFileTestData> {};
struct apiwmtx : public TestWithParam<APIFileTestData> {};
struct apitns : public TestWithParam<APIFileTestData> {};

TEST_P(apiset, api) {
  Tensor<double> tensor = GetParam().getTensor();
  SCOPED_TRACE("Tensor name " + tensor.getName());

  auto storage = tensor.getStorage();

  auto& expectedIndices = GetParam().getExpectedIndices();
  auto& expectedValues = GetParam().getExpectedValues();
  ASSERT_COMPONENTS_EQUALS(expectedIndices, expectedValues, tensor);
}

TEST_P(apiget, api) {
  TensorBase tensor = GetParam().getTensor();

  auto format = tensor.getFormat();
  taco_iassert(format == taco::CSR || format == taco::CSC);

  auto storage = tensor.getStorage();
  auto index = storage.getIndex();

  ASSERT_ARRAY_EQ(GetParam().getExpectedIndices()[1][0],
                  {(int*)index.getModeIndex(1).getIndexArray(0).getData(),
                   index.getModeIndex(1).getIndexArray(0).getSize()});
  ASSERT_ARRAY_EQ(GetParam().getExpectedIndices()[1][1],
                  {(int*)index.getModeIndex(1).getIndexArray(1).getData(),
                   index.getModeIndex(1).getIndexArray(1).getSize()});

  ASSERT_ARRAY_EQ(GetParam().getExpectedValues(),
                  {(double*)storage.getValues().getData(),
                   storage.getIndex().getSize()});
}

TEST_P(apiwrb, api) {
  TensorBase tensor = GetParam().getTensor(CSC);

  auto storage = tensor.getStorage();
  if (tensor.getFormat() == taco::CSC) {
    std::string testdir = std::string("\"") + testDirectory() + "\"";
    auto tmpdir = util::getTmpdir();
    std::string datafilename=testdir + "/data/" + GetParam().getFilename();
    std::string CSCfilename=tmpdir + GetParam().getFilename() + ".csc";

    write(CSCfilename, FileType::rb, tensor);

    std::string diffcommand="diff -wB <(tail -n +3 " + CSCfilename
        + " ) <(tail -n +3 " + datafilename + " ) > diffresult ";
    std::ofstream diffcommandfile;
    diffcommandfile.open("diffcommand.tac");
    diffcommandfile << diffcommand.c_str();
    diffcommandfile.close();
    ASSERT_FALSE(system("chmod +x diffcommand.tac ; bash ./diffcommand.tac "));
    std::ifstream diffresult("diffresult");
    bool nodiff=(diffresult.peek() == std::ifstream::traits_type::eof());
    std::string cleancommand="rm diffresult diffcommand.tac "+CSCfilename;
    ASSERT_FALSE(system(cleancommand.c_str()));
    ASSERT_TRUE(nodiff);
  }
}

TEST_P(apiwmtx, api) {
  TensorBase tensor = GetParam().getTensor(CSC);
  tensor.pack();

  auto storage = tensor.getStorage();

  std::string extension;
  if (isDense(tensor.getFormat()))
    extension = ".ttx";
  else
    extension = ".mtx";
  std::string testdir = std::string("\"") + testDirectory() + "\"";
  auto tmpdir = util::getTmpdir();
  std::string datafilename = testdir + "/data/"
                           + GetParam().getFilename() + extension;
  std::string filename = tmpdir + GetParam().getFilename() + ".test";

  write(filename, FileType::mtx, tensor);

  string diffresultfile = tmpdir + "diffresult";
  string diffcommand = "diff -wB -I '^%.*' " + filename + " " +
      datafilename + " > " + diffresultfile;
  string diffcommandfile = tmpdir + "diffcommand.tac";
  std::ofstream diffcommandstream;
  diffcommandstream.open(diffcommandfile);
  diffcommandstream << diffcommand.c_str();
  diffcommandstream.close();
  ASSERT_FALSE(system(("chmod +x " + diffcommandfile + " ; bash " +
      diffcommandfile).c_str()));
  std::ifstream diffresult(diffcommand);
  bool nodiff=(diffresult.peek() == std::ifstream::traits_type::eof());
  string cleancommand = "rm " + diffresultfile + " " + diffcommandfile + " " +
      filename;
  ASSERT_FALSE(system(cleancommand.c_str()));
  ASSERT_TRUE(nodiff);
}

TEST_P(apitns, api) {
  TensorBase tensor = GetParam().getTensor(Sparse);
  tensor.pack();

  const std::string tmpdir = util::getTmpdir();
  const std::string filename = tmpdir + GetParam().getFilename();
  write(filename, FileType::tns, tensor);

  TensorBase newTensor = read(filename, tensor.getFormat());
  ASSERT_TRUE(equals(tensor, newTensor));
}

INSTANTIATE_TEST_CASE_P(load, apiset, Values(
  APIMatrixStorageTestData(d33a_CSR("A"),
    {
      {
        // Dense index
        {3}
      },
      {
        // Sparse index
        {0, 1, 1, 3},
        {1, 0, 2},
      }
    },
      {2, 3, 4}
  ),
  APIMatrixStorageTestData(d33a_CSC("A"),
    {
      {
        // Dense index
        {3}
      },
      {
        // Sparse index
        {0, 1, 2, 3},
        {2, 0, 2},
      },
    },
    {3, 2, 4}
  ),
  APIMatrixStorageTestData(d35a_CSR("A"),
    {
      {
        // Dense index
        {3}
      },
      {
        // Sparse index
        {0, 2, 2, 4},
        {0, 1, 0, 3},
      }
    },
    {2, 4, 3, 5}
  ),
  APIMatrixStorageTestData(d35a_CSC("A"),
    {
      {
        // Dense index
	{5}
      },
      {
        // Sparse index
        {0, 2, 3, 3, 4, 4},
        {0, 2, 0, 2},
      }
    },
    {2, 3, 4, 5}
  ),
  APIMatrixStorageTestData("rua_32.mtx",
    {
      {
        // Dense index
        {32}
      },
      {
        // Sparse index
        {0, 6, 11, 17, 21, 25, 28, 33, 38, 45, 52, 57, 60, 62, 64, 67,
         70, 73, 78, 81, 84, 87, 89, 93, 96, 101, 105, 109, 111, 116, 120, 123,
         126},
        {0, 1, 2, 3, 6, 25, 0, 1, 8, 20, 27, 1, 2, 5, 7, 8,
         28, 2, 3, 4, 11, 2, 4, 22, 26, 0, 5, 15, 2, 6, 13,
         20, 30, 0, 7, 11, 16, 26, 6, 8, 9, 12, 18, 22, 26,
         0, 9, 10, 20, 22, 24, 26, 1, 10, 14, 17, 28, 5, 11,
         23, 10, 12, 2, 13, 1, 14, 19, 3, 15, 21, 3, 15, 16,
         5, 9, 17, 19, 29, 0, 18, 25, 7, 15, 19, 2, 20, 31,
         10, 21, 1, 16, 20, 22, 11, 23, 25, 5, 14, 17, 23, 24,
         12, 17, 21, 25, 4, 23, 25, 26, 8, 27, 2, 4, 26, 28,
         31, 11, 16, 22, 29, 12, 13, 30, 23, 27, 31}
      }
   },
{101.0, 102.0, 103.0, 104.0, 107.0, 126.0, 201.0, 202.0, 209.0, 221.0,
 228.0, 302.0, 303.0, 306.0, 308.0, 309.0, 329.0, 403.0, 404.0, 405.0,
 412.0, 503.0, 505.0, 523.0, 527.0, 601.0, 606.0, 616.0, 703.0, 707.0,
 714.0, 721.0, 731.0, 801.0, 808.0, 812.0, 817.0, 827.0, 907.0, 909.0,
 910.0, 913.0, 919.0, 923.0, 927.0, 1001.0, 1010.0, 1011.0, 1021.0, 1023.0,
 1025.0, 1027.0, 1102.0, 1111.0, 1115.0, 1118.0, 1129.0, 1206.0, 1212.0, 1224.0,
 1311.0, 1313.0, 1403.0, 1414.0, 1502.0, 1515.0, 1520.0, 1604.0, 1616.0, 1622.0,
 1704.0, 1716.0, 1717.0, 1806.0, 1810.0, 1818.0, 1820.0, 1830.0, 1901.0, 1919.0,
 1926.0, 2008.0, 2016.0, 2020.0, 2103.0, 2121.0, 2132.0, 2211.0, 2222.0, 2302.0,
 2317.0, 2321.0, 2323.0, 2412.0, 2424.0, 2426.0, 2506.0, 2515.0, 2518.0, 2524.0,
 2525.0, 2613.0, 2618.0, 2622.0, 2626.0, 2705.0, 2724.0, 2726.0, 2727.0, 2809.0,
 2828.0, 2903.0, 2905.0, 2927.0, 2929.0, 2932.0, 3012.0, 3017.0, 3023.0, 3030.0,
 3113.0, 3114.0, 3131.0, 3224.0, 3228.0, 3232.0}
  ),
  APIMatrixStorageTestData("d33.mtx",
    {
      {
        // Dense index
        {3}
      },
      {
        // Sparse index
        {0, 1, 2, 3},
        {2, 0, 2},
      },
   },
   {3, 2, 4}
  )
)
);

INSTANTIATE_TEST_CASE_P(write, apiget, Values(
  APIMatrixStorageTestData(d33a_CSR("A"),
    {
      {
        // Dense index
        {3}
      },
      {
        // Sparse index
        {0, 1, 1, 3},
        {1, 0, 2},
      }
    },
    {2, 3, 4}
  ),
  APIMatrixStorageTestData(d33a_CSC("A"),
    {
      {
        // Dense index
        {3}
      },
      {
        // Sparse index
        {0, 1, 2, 3},
        {2, 0, 2},
      },
    },
    {3, 2, 4}
  ),
  APIMatrixStorageTestData("rua_32.mtx",
    {
      {
        // Dense index
        {32}
      },
      {
        // Sparse index
        {0, 6, 11, 17, 21, 25, 28, 33, 38, 45, 52, 57, 60, 62, 64, 67,
         70, 73, 78, 81, 84, 87, 89, 93, 96, 101, 105, 109, 111, 116, 120, 123,
         126},
        {0, 1, 2, 3, 6, 25, 0, 1, 8, 20, 27, 1, 2, 5, 7, 8,
         28, 2, 3, 4, 11, 2, 4, 22, 26, 0, 5, 15, 2, 6, 13,
         20, 30, 0, 7, 11, 16, 26, 6, 8, 9, 12, 18, 22, 26,
         0, 9, 10, 20, 22, 24, 26, 1, 10, 14, 17, 28, 5, 11,
         23, 10, 12, 2, 13, 1, 14, 19, 3, 15, 21, 3, 15, 16,
         5, 9, 17, 19, 29, 0, 18, 25, 7, 15, 19, 2, 20, 31,
         10, 21, 1, 16, 20, 22, 11, 23, 25, 5, 14, 17, 23, 24,
         12, 17, 21, 25, 4, 23, 25, 26, 8, 27, 2, 4, 26, 28,
         31, 11, 16, 22, 29, 12, 13, 30, 23, 27, 31}
      }
    },
{101.0, 102.0, 103.0, 104.0, 107.0, 126.0, 201.0, 202.0, 209.0, 221.0,
 228.0, 302.0, 303.0, 306.0, 308.0, 309.0, 329.0, 403.0, 404.0, 405.0,
 412.0, 503.0, 505.0, 523.0, 527.0, 601.0, 606.0, 616.0, 703.0, 707.0,
 714.0, 721.0, 731.0, 801.0, 808.0, 812.0, 817.0, 827.0, 907.0, 909.0,
 910.0, 913.0, 919.0, 923.0, 927.0, 1001.0, 1010.0, 1011.0, 1021.0, 1023.0,
 1025.0, 1027.0, 1102.0, 1111.0, 1115.0, 1118.0, 1129.0, 1206.0, 1212.0, 1224.0,
 1311.0, 1313.0, 1403.0, 1414.0, 1502.0, 1515.0, 1520.0, 1604.0, 1616.0, 1622.0,
 1704.0, 1716.0, 1717.0, 1806.0, 1810.0, 1818.0, 1820.0, 1830.0, 1901.0, 1919.0,
 1926.0, 2008.0, 2016.0, 2020.0, 2103.0, 2121.0, 2132.0, 2211.0, 2222.0, 2302.0,
 2317.0, 2321.0, 2323.0, 2412.0, 2424.0, 2426.0, 2506.0, 2515.0, 2518.0, 2524.0,
 2525.0, 2613.0, 2618.0, 2622.0, 2626.0, 2705.0, 2724.0, 2726.0, 2727.0, 2809.0,
 2828.0, 2903.0, 2905.0, 2927.0, 2929.0, 2932.0, 3012.0, 3017.0, 3023.0, 3030.0,
 3113.0, 3114.0, 3131.0, 3224.0, 3228.0, 3232.0}
  )
)
);

INSTANTIATE_TEST_CASE_P(write, apiwrb,
  Values(
      APIFileTestData("rua_32.rb")
  )
);

INSTANTIATE_TEST_CASE_P(write, apiwmtx,
  Values(
      APIFileTestData("d33", CSC),
      APIFileTestData("rua_32", CSC),
      APIFileTestData("d33",Format({Dense,Dense}))
  )
);

INSTANTIATE_TEST_CASE_P(readwrite, apitns,
  Values(
    APIFileTestData(d5d("d", Format({Sparse})), "d5d.tns"),
    APIFileTestData(d233c("c", Format({Sparse, Sparse, Sparse})), "d233c.tns")
  )
);
