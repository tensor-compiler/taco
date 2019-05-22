#include "taco/parser/einsum_parser.h"
#include "taco/parser/parser.h"
#include "taco/util/name_generator.h"
#include "taco/util/strings.h"
#include "taco/tensor.h"


#include <algorithm>

namespace taco{
namespace parser{

EinsumParser::EinsumParser(const std::string &expression, const std::vector<TensorBase> &tensors) {

  if(expression.empty()){
    throw ParseError("No input operands");
  }

  // Remove spaces from expression
  std::string subscripts;
  std::remove_copy(expression.begin(), expression.end(), std::back_inserter(subscripts), ' ');
//  tensorExpressions = parseToTaco(expression, tensors);
}

//void EinsumParser::parse(){
//
//}
//
//TensorBase EinsumParser::getResultTensor(){
//  return resultTensor;
//}

std::vector<std::string> EinsumParser::genUnusedSymbols(std::set<std::string> &usedSymbols, int numUnusedSymbolsNeeded) {
  const std::string baseEinsumChars("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

  // Base index name to use if we run out of einsum characters.
  const std::string baseName("ein");

  int unusedSymbolsFound = 0, currentIdx = 0;
  std::vector<std::string> unusedSymbols;
  taco::util::NameGenerator nameGenerator({baseName});

  // First consume einsum characters
  while(unusedSymbolsFound < numUnusedSymbolsNeeded && currentIdx < (int)baseEinsumChars.length()){

    std::string currentSymbol(1, baseEinsumChars[currentIdx++]);
    if(usedSymbols.count(currentSymbol) == 0){
      unusedSymbols.emplace_back(currentSymbol);
      unusedSymbolsFound++;
    }
  }

  // If we run out of einsum variables, create new variables as placeholders.
  while(unusedSymbolsFound < numUnusedSymbolsNeeded){
    unusedSymbols.push_back(nameGenerator.getUniqueName(baseName));
    unusedSymbolsFound++;
  }

  return unusedSymbols;
}

std::string EinsumParser::findOutputString(const std::string &subscripts){
  std::string tempString;
  std::remove_copy_if(subscripts.begin(), subscripts.end(), std::back_inserter(tempString),
                      [](char chr){ return chr == '.' || chr == ',';});

  std::sort(tempString.begin(), tempString.end());

  // Count number of times each index appears since einstein notation implicitly reduces duplicated indices
  std::map<char, int> letterCount;
  for(const char &letter : tempString){
    if(letterCount.count(letter) == 1){
      letterCount[letter]++;
    }else{
      letterCount[letter] = 1;
    }
  }

  // Only keep subscripts that appear once
  std::string outputString;
  for(const char &letter : tempString){
    if(letterCount[letter] == 1){
      outputString.push_back(letter);
    }
  }

  return outputString;
}

bool EinsumParser::exprHasOutput(const std::string &subscripts){
  // Go through subscripts and check for ->
  int dash_count = 0, gt_count = 0;
  for(int i = 0; i < (int)subscripts.size(); i++){
    dash_count = subscripts[i] == '-'? dash_count + 1: dash_count;
    gt_count = subscripts[i] == '>'? gt_count + 1: gt_count;

    if(subscripts[i] == '-' && (i + 1) < (int)subscripts.size() && subscripts[i+1] != '>'){
      throw ParseError("Subscripts must contain '->' instead of '-'.");
    }
  }

  if(dash_count != gt_count || dash_count > 1){
    throw ParseError("Subscripts may only contain one '->'.");
  }

  return dash_count == 1;
}

std::string EinsumParser::convertToIndexExpr(const std::string &subscripts, const std::vector<std::string> &ellipsisReplacement,
                                             const std::string &tensorName) {

  std::string tensorString(tensorName + "(");

  int ellipseCount = 0;

  for(int i = 0; i < (int)subscripts.length(); ++i){
    const char subscript = subscripts[i];
    if(subscript == '.'){
      ellipseCount++;
      if(ellipseCount == 3){
        if(subscripts[i - 2] != '.' || subscripts[i-1] != '.'){
          throw ParseError("Ellipses must be consecutive");
        }
        for(const auto& replacement: ellipsisReplacement){
          tensorString.append(replacement);
          tensorString.push_back(',');
        }
      }
    }else{
      tensorString.push_back(subscript);
      tensorString.push_back(',');
    }
  }

  if(ellipseCount != 0 && ellipseCount != 3){
    throw ParseError("Incorrect number of ellipses present in index expression.");
  }

  if(tensorString[tensorString.length() - 1] == ','){
    tensorString[tensorString.length() - 1] = ')';
  }else{
    tensorString.push_back(')');
  }

  // Need to check if we actually need a scalar
  if(tensorString[tensorString.length() - 2] == '(' && tensorString[tensorString.length() - 1] == ')'){
    return tensorName;
  }

  return tensorString;
}

std::vector<std::string> EinsumParser::parseToTaco(const std::string &subscripts,
                                                   const std::vector<TensorBase> &tensors) {

  // Split operands list and get output operand
  bool hasOutput = exprHasOutput(subscripts);
  std::vector<std::string> inputTensorSubscripts;
  std::string outputSubscripts, unsplitInputs;
  if(hasOutput) {
    std::vector<std::string> temp = taco::util::split(subscripts, "->");
    outputSubscripts = temp.size() == 2? temp[1]: outputSubscripts;
    unsplitInputs = temp[0];
    inputTensorSubscripts = taco::util::split(unsplitInputs, ",");
  }else {
    inputTensorSubscripts = taco::util::split(subscripts, ",");
    unsplitInputs = subscripts;
    outputSubscripts = "..." + findOutputString(subscripts);
  }

  // Check that all output subscripts are in the input
  for(const char &sub : outputSubscripts){
    if(sub != '.' && unsplitInputs.find(sub) == std::string::npos){
      throw ParseError("Output contains an index not in the input.");
    }
  }

  // Compute set of used characters
  std::set<std::string> usedSymbols;
  for(const char &subscript : subscripts){
    if(isalpha(subscript)){
      usedSymbols.emplace(1, subscript);
    }
  }

  // Compute the max size of the tensor
  int maxTensorOrder = 0;
  for(const auto &tensor : tensors){
    maxTensorOrder = std::max(maxTensorOrder, tensor.getOrder());
  }

  // Create a list of unused index names
  std::vector<std::string> unusedSymbols = genUnusedSymbols(usedSymbols, maxTensorOrder);

  int longestOmittedDimSize = 0;
  std::vector<std::string> tensorsExprs;

  for(int currentTensor = 0; currentTensor < (int)tensors.size(); ++currentTensor){
    std::string &tensorSubscript = inputTensorSubscripts[currentTensor];

    if(tensorSubscript.find('.') != std::string::npos){

      int dimsRemaining = tensors[currentTensor].getOrder() - (int)(tensorSubscript.length() - 3);
      longestOmittedDimSize = std::max(longestOmittedDimSize, dimsRemaining);

      if(dimsRemaining < 0){
        throw ParseError("Ellipses lengths do not match.");
      }else{
        std::vector<std::string> tacoIndices(unusedSymbols.end() - dimsRemaining, unusedSymbols.end());
        tensorsExprs.push_back(convertToIndexExpr(tensorSubscript, tacoIndices, tensors[currentTensor].getName()));
      }

    }else{
      tensorsExprs.push_back(convertToIndexExpr(tensorSubscript, {""}, tensors[currentTensor].getName()));
    }
  }

  std::string tacoOutput;

  // Index vars to replace ellipses in output if any
  std::vector<std::string> outEllipseIndices(unusedSymbols.end() - longestOmittedDimSize, unusedSymbols.end());
  tacoOutput = convertToIndexExpr(outputSubscripts, outEllipseIndices, "out");

  // Ensure number of tensors equals the number of terms
  if(tensorsExprs.size() != tensors.size()){
    throw ParseError("Number of subscripts must be equal to the number of terms.");
  }

  tensorsExprs.push_back(tacoOutput);
  return tensorsExprs;
}

}}
