#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "parser/parser.h"
#include "tensor.h"
#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "error.h"
#include "util/strings.h"

using namespace std;
using namespace taco;

static void printFlag(string flag, string text) {
  const size_t descriptionStart = 15;
  const size_t columnEnd        = 80;
  string flagString = "  -" + flag +
                      util::repeat(" ",descriptionStart-(flag.size()+3));
  cout << flagString;
  size_t column = flagString.size();
  vector<string> words = util::split(text, " ");
  for (auto& word : words) {
    if (column + word.size()+1 >= columnEnd) {
      cout << endl << util::repeat(" ", descriptionStart);
      column = descriptionStart;
    }
    column += word.size()+1;
    cout << word << " ";
  }
  cout << endl;
}

static void printUsageInfo() {
  cout << "Usage: taco [options] <index expression>" << endl;
  cout << endl;
  cout << "Options:" << endl;
  printFlag("f=<format>",
            "Specify the format of a tensor in the expression. Formats are "
            "specified per dimension using d (dense) and s (sparse). "
            "All formats default to dense. "
            "Examples: A:ds, b:d and D:sss.");
  cout << endl;
  cout << "Options planned for the future:" << endl;
  printFlag("c", "Print compute IR (default).");
  cout << endl;
  printFlag("a", "Print assembly IR.");
  cout << endl;
  printFlag("g",
            "Generate random data for a given tensor. (e.g. B).");
  cout << endl;
  printFlag("i",
            "Initialize a tensor from an input file (e.g. B:\"myfile.txt\"). "
            "If all the tensors have been initialized then the expression is "
            "evaluated.");
  cout << endl;
  printFlag("o",
            "Write the result of evaluating the expression to the given file");
  cout << endl;
  printFlag("t", "Time compilation, assembly and computation.");
}

static int reportError(string errorMessage, int errorCode) {
  cerr << "Error: " << errorMessage << endl << endl;
  printUsageInfo();
  return errorCode;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printUsageInfo();
    return 1;
  }

  string exprStr;
  map<string,Format> formats;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if ("-f=" == arg.substr(0,3)) {
      vector<string> descriptor = util::split(arg.substr(3,string::npos), ":");
      if (descriptor.size() != 2) {
        return reportError("Incorrect format descriptor", 3);
      }
      string tensorName   = descriptor[0];
      string formatString = descriptor[1];
      Format::LevelTypes      levelTypes;
      Format::DimensionOrders dimensions;
      for (size_t i = 0; i < formatString.size(); i++) {
        switch (formatString[i]) {
          case 'd':
            levelTypes.push_back(LevelType::Dense);
            break;
          case 's':
            levelTypes.push_back(LevelType::Sparse);
            break;
          default:
            return reportError("Incorrect format descriptor", 3);
            break;
        }
        dimensions.push_back(i);
      }
      formats.insert({tensorName, Format(levelTypes, dimensions)});
    }
    else {
      if (exprStr.size() != 0) {
        printUsageInfo();
        return 2;
      }
      exprStr = argv[i];
    }
  }

  internal::Tensor tensor;
  try {
    parser::Parser parser(exprStr, formats);
    parser.parse();
    tensor = parser.getResultTensor();
  } catch (parser::ParseError& e) {
    cerr << e.getMessage() << endl;
  }

  tensor.compile();
  tensor.printComputeIR(cout);

  return 0;
}
