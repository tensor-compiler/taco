#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "taco/tensor_base.h"
#include "taco/expr.h"
#include "taco/operator.h"
#include "taco/parser/parser.h"
#include "taco/storage/storage.h"

#include "ir/ir.h"
#include "lower/lower_codegen.h"
#include "lower/iterators.h"
#include "lower/iteration_schedule.h"
#include "lower/merge_lattice.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"
#include "taco/util/benchmark.h"
#include "taco/util/fill.h"

using namespace std;
using namespace taco;

#define TOOL_BENCHMARK(CODE,NAME,REPEAT) {        \
    if (time) {                                   \
      TACO_BENCHMARK(CODE,REPEAT,timevalue);      \
      std::cout << NAME << " Time " << timevalue; \
    }                                             \
    else {                                        \
      CODE;                                       \
    }                                             \
}

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
  cout << "Examples:" << endl;
  cout << "  taco \"a(i) = b(i) + c(i)\"                            # Dense vector add" << endl;
  cout << "  taco \"a(i) = b(i) + c(i)\" -f=b:s -f=c:s -f=a:s       # Sparse vector add" << endl;
  cout << "  taco \"a(i) = B(i,j) * c(j)\" -f=B:ds                  # SpMV" << endl;
  cout << "  taco \"A(i,l) = B(i,j,k) * C(j,l) * D(k,l)\" -f=B:sss  # MTTKRP" << endl;
  cout << endl;
  cout << "Options:" << endl;
  printFlag("f=<format>",
            "Specify the format of a tensor in the expression. Formats are "
            "specified per dimension using d (dense) and s (sparse). "
            "All formats default to dense. "
            "Examples: A:ds, b:d and D:sss.");
  cout << endl;
  printFlag("c", "Print compute IR (default).");
  cout << endl;
  printFlag("a", "Print assembly IR.");
  cout << endl;
  printFlag("l=<var>",
            "Print merge lattice IR for the given index variable.");
  cout << endl;
  printFlag("g=<fill>",
            "Generate random data for a given tensor. Vectors can be d "
            "(dense), s (sparse) or h (hypersparse). Matrices can be d, s, h or"
            " l (slicing), f (FEM), b (Blocked).");
  cout << endl;
  printFlag("r=<read>",
            "Read a matrix from file in HB or MTX file format.");
  cout << endl;
  printFlag("nocolor", "Print without colors.");
  cout << endl;
  printFlag("t=<repeat>", "Time compilation, assembly and "
      "<repeat> times computation.");
  cout << endl;

  /*
  cout << "Options planned for the future:" << endl;
  printFlag("i",
            "Initialize a tensor from an input file (e.g. B:\"myfile.txt\"). "
            "If all the tensors have been initialized then the expression is "
            "evaluated.");
  cout << endl;
  printFlag("o",
            "Write the result of evaluating the expression to the given file");
  cout << endl;
  */
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

  bool printCompute  = false;
  bool printAssemble = false;
  bool printLattice = false;
  bool evaluate = false;
  bool color = true;
  bool time = false;
  int  repeat = 1;

  taco::util::timeResults timevalue;

  string indexVarName = "";

  string exprStr;
  map<string,Format> formats;
  map<string,int> tensorsSize;
  map<string,taco::util::FillMethod> tensorsFill;
  map<string,string> tensorsFileNames;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if ("-f=" == arg.substr(0,3)) {
      vector<string> descriptor = util::split(arg.substr(3,string::npos), ":");
      if (descriptor.size() < 2 || descriptor.size() > 3) {
        return reportError("Incorrect format descriptor", 3);
      }
      string tensorName     = descriptor[0];
      string formatString   = descriptor[1];
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
      if (descriptor.size() > 2) {
        string dimOrderString = descriptor[2];
        dimensions.clear();
        for (size_t i = 0; i < dimOrderString.size(); i++) {
          dimensions.push_back(dimOrderString[i] - '0');
        }
      }
      formats.insert({tensorName, Format(levelTypes, dimensions)});
    }
    else if ("-g=" == arg.substr(0,3)) {
      vector<string> descriptor = util::split(arg.substr(3,string::npos), ":");
      if (descriptor.size() < 2 || descriptor.size() > 3) {
        return reportError("Incorrect generating descriptor", 3);
      }
      string tensorName   = descriptor[0];
      std::vector<taco::util::FillMethod> fillMethods;
      string fillString   = descriptor[1];
      string genoptions = descriptor[2];
      if (genoptions.size()>1) {
        int tensorDim = std::stoi(genoptions);
        tensorsSize.insert({tensorName, tensorDim});
      }
      switch (fillString[0]) {
        case 'd': {
          tensorsFill.insert({tensorName, taco::util::FillMethod::Dense});
          break;
        }
        case 's': {
          tensorsFill.insert({tensorName, taco::util::FillMethod::Sparse});
          break;
        }
        case 'h': {
          tensorsFill.insert({tensorName, taco::util::FillMethod::HyperSpace});
          break;
        }
        case 'l': {
          tensorsFill.insert({tensorName, taco::util::FillMethod::Slicing});
          break;
        }
        case 'f': {
          tensorsFill.insert({tensorName, taco::util::FillMethod::FEM});
          break;
        }
        case 'b': {
          tensorsFill.insert({tensorName, taco::util::FillMethod::Blocked});
          break;
        }
        default: {
          return reportError("Incorrect generating descriptor", 3);
          break;
        }
      }
      evaluate = true;
    }
    else if ("-r=" == arg.substr(0,3)) {
          vector<string> descriptor = util::split(arg.substr(3,string::npos), ":");
          if (descriptor.size() < 3) {
            return reportError("Incorrect read descriptor", 3);
          }
          string tensorName     = descriptor[0];
          int tensorDim = std::stoi(descriptor[1]);
          tensorsSize.insert({tensorName, tensorDim});
          string fileName       = descriptor[2];
          tensorsFileNames.insert({tensorName,fileName});
          evaluate = true;
    }
    else if ("-c" == arg.substr(0,2)) {
      printCompute = true;
    }
    else if ("-a" == arg.substr(0,2)) {
      printAssemble = true;
    }
    else if ("-l=" == arg.substr(0,3)) {
      indexVarName = arg.substr(3,string::npos);
      printLattice = true;
    }
    else if ("-nocolor" == arg) {
      color = false;
    }
    else if ("-t=" == arg.substr(0,3)) {
      time = true;
      try {
        repeat=stoi(arg.substr(3,string::npos));
      }
      catch (...) {
        return reportError("Incorrect time descriptor", 3);
      }
    }
    else {
      if (exprStr.size() != 0) {
        printUsageInfo();
        return 2;
      }
      exprStr = argv[i];
    }
  }

  // Print compute is the default if nothing else was asked for
  if (!printAssemble && !printLattice && !evaluate) {
    printCompute = true;
  }

  TensorBase tensor;
  parser::Parser parser(exprStr, formats, tensorsSize);
  try {
    parser.parse();
    tensor = parser.getResultTensor();
  } catch (parser::ParseError& e) {
    cerr << e.getMessage() << endl;
  }

  if (printLattice && !parser.hasIndexVar(indexVarName)) {
    return reportError("Index variable is not in expression", 4);
  }

  if (!evaluate) {
    TOOL_BENCHMARK(tensor.compile(),"Compile",1);
  }

  string gentext = "Generated by the Tensor Algebra Compiler (tensor-compiler.org)";
  std::string green = (color) ? "\033[38;5;70m" : "";
  std::string nc    = (color) ? "\033[0m"       : "";
  cout << green << "/* " << gentext << " */" << nc << endl;

  bool hasPrinted = false;
  if (printAssemble) {
    tensor.printAssemblyIR(cout,color);
    hasPrinted = true;
  }

  if (printCompute) {
    if (hasPrinted) {
      cout << endl << endl;
    }
    tensor.printComputeIR(cout,color);
    hasPrinted = true;
  }

  if (printLattice) {
    if (hasPrinted) {
      cout << endl << endl;
    }
    taco::Var indexVar = parser.getIndexVar(indexVarName);
    lower::IterationSchedule schedule = lower::IterationSchedule::make(tensor);
    map<TensorBase,ir::Expr> tensorVars;
    tie(std::ignore, std::ignore, tensorVars) = lower::getTensorVars(tensor);
    lower::Iterators iterators(schedule, tensorVars);
    auto lattice = lower::MergeLattice::make(tensor.getExpr(), indexVar,
                                             schedule, iterators);
    cout << lattice;
    hasPrinted = true;
  }
  cout << endl;

  if (evaluate) {
    TensorBase paramTensor;
    for ( const auto &fills : tensorsFill ) {
      paramTensor = parser.getTensor(fills.first);
      taco::util::fillTensor(paramTensor, fills.second);
    }
    for ( const auto &loads : tensorsFileNames ) {
      paramTensor = parser.getTensor(loads.first);
      paramTensor.read(loads.second);
    }
    TOOL_BENCHMARK(tensor.compile(),"Compile",1);
    TOOL_BENCHMARK(tensor.assemble(),"Assemble",1);
    TOOL_BENCHMARK(tensor.compute(),"Compute",repeat);
    std::cout << "Result " << tensor << std::endl;
  }

  return 0;
}
