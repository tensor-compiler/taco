#include "test.h"
#include "test_tensors.h"

#include "expr_factory.h"

#include "format.h"
#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "expr_nodes.h"
#include "storage/storage.h"
#include "operator.h"

#include <cmath>
#include <stdio.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

using namespace taco;

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

struct APIStorage {
	APIStorage(Tensor<double> tensor,
           const Indices& expectedIndices,
           const vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }

  Tensor<double> tensor;
  Indices        expectedIndices;
  vector<double> expectedValues;
};

struct APIFile {
	APIFile(Tensor<double> tensor,
		std::string filename)
	: tensor(tensor),
	  filename(filename) {
	}
	Tensor<double> tensor;
	std::string filename;
};

struct apiset : public TestWithParam<APIStorage> {};
struct apiget : public TestWithParam<APIStorage> {};
struct apiw : public TestWithParam<APIFile> {};

TEST_P(apiset, api) {
  Tensor<double> tensor = GetParam().tensor;

  auto storage = tensor.getStorage();
  ASSERT_TRUE(storage.defined());
  auto levels = storage.getFormat().getLevels();

  // Check that the indices are as expected
  auto& expectedIndices = GetParam().expectedIndices;
  auto size = storage.getSize();

  for (size_t i=0; i < levels.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto levelIndex = storage.getLevelIndex(i);
    auto levelIndexSize = size.levelIndices[i];

    switch (levels[i].getType()) {
      case LevelType::Dense: {
        iassert(expectedIndex.size() == 1);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, levelIndexSize.ptr});
        ASSERT_EQ(nullptr, levelIndex.idx);
        ASSERT_EQ(0u, levelIndexSize.idx);
        break;
      }
      case LevelType::Sparse: {
        iassert(expectedIndex.size() == 2);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, levelIndexSize.ptr});
        ASSERT_ARRAY_EQ(expectedIndex[1], {levelIndex.idx, levelIndexSize.idx});
        break;
      }
      case LevelType::Fixed:
        break;
    }
  }

  auto& expectedValues = GetParam().expectedValues;
  ASSERT_EQ(expectedValues.size(), storage.getSize().values);
  ASSERT_ARRAY_EQ(expectedValues, {storage.getValues(), size.values});
}

TEST_P(apiget, api) {
  Tensor<double> tensor = GetParam().tensor;

  auto storage = tensor.getStorage();
  ASSERT_TRUE(storage.defined());

  // Check that the indices are as expected
  auto& expectedIndices = GetParam().expectedIndices;
  auto size = storage.getSize();

  double* A;
  int* IA;
  int* JA;
  if (tensor.getFormat().isCSR()) {
    tensor.getCSR(A,IA,JA);
    auto& expectedValues = GetParam().expectedValues;
    ASSERT_ARRAY_EQ(expectedValues, {A,size.values});
    ASSERT_ARRAY_EQ(expectedIndices[1][0], {IA, size.levelIndices[1].ptr});
    ASSERT_ARRAY_EQ(expectedIndices[1][1], {JA, size.levelIndices[1].idx});
  }
  if (tensor.getFormat().isCSC()) {
    tensor.getCSC(A,IA,JA);
    auto& expectedValues = GetParam().expectedValues;
    ASSERT_ARRAY_EQ(expectedValues, {A,size.values});
    ASSERT_ARRAY_EQ(expectedIndices[1][0], {IA, size.levelIndices[1].ptr});
    ASSERT_ARRAY_EQ(expectedIndices[1][1], {JA, size.levelIndices[1].idx});
  }
}

TEST_P(apiw, api) {
  Tensor<double> tensor = GetParam().tensor;

  auto storage = tensor.getStorage();
  ASSERT_TRUE(storage.defined());
  auto size = storage.getSize();

  if (tensor.getFormat().isCSC()) {
    std::string testdir=TOSTRING(TACO_TEST_DIR);
    std::string datafilename=testdir + "/data/" + GetParam().filename;
    std::string CSCfilename=GetParam().filename+".csc";
    tensor.writeHB(CSCfilename);
    std::string diffcommand="diff -wB <(tail -n +3 " + CSCfilename + " ) <(tail -n +3 " + datafilename + " ) > diffresult ";
    std::ofstream diffcommandfile;
    diffcommandfile.open("diffcommand.tac");
    diffcommandfile << diffcommand.c_str();
    diffcommandfile.close();
    system("chmod +x diffcommand.tac ; bash ./diffcommand.tac ");
    std::ifstream diffresult("diffresult");
    bool nodiff=(diffresult.peek() == std::ifstream::traits_type::eof());
//    std::string cleancommand="rm diffresult diffcommand.tac "+CSCfilename;
//    system(cleancommand.c_str());
    ASSERT_TRUE(nodiff);
  }
}

INSTANTIATE_TEST_CASE_P(load, apiset,
  Values(
      APIStorage(d33a_CSR("A"),
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
      APIStorage(d33a_CSC("A"),
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
      APIStorage(d35a_CSR("A"),
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
      APIStorage(d35a_CSC("A"),
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
      APIStorage(rua32("RUA_32"),
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
	APIStorage(d33a_MTX("A"),
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

INSTANTIATE_TEST_CASE_P(write, apiget,
  Values(
      APIStorage(d33a_CSR("A"),
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
      APIStorage(d33a_CSC("A"),
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
      APIStorage(rua32("RUA_32"),
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

INSTANTIATE_TEST_CASE_P(write, apiw,
  Values(
      APIFile(rua32("RUA_32"),"rua_32.rb")
  )
);

