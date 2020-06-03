#include "taco/parser/einsum_parser.h"
#include "taco/parser/parser.h"
#include "taco/util/name_generator.h"
#include "taco/util/strings.h"
#include "taco/tensor.h"

#include <algorithm>

namespace taco{
namespace parser{

EinsumParser::EinsumParser(const std::string &expression,
                           std::vector<TensorBase> &tensors,
                           Format &format,
                           Datatype outType) : outType(outType), format(format), tensors(tensors) {

  einsumSymbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  einSumSymbolsSet = std::set<char>(einsumSymbols.begin(), einsumSymbols.end());
  einsumPunctuation = ".,->";

  if(expression.empty()){
    throw ParseError("No input operands");
  }

  // Remove spaces from expression
  std::remove_copy(expression.begin(), expression.end(), std::back_inserter(subscripts), ' ');

  // Check that all elements are valid and throw error otherwise
  for(const auto& s: subscripts) {
    if(einSumSymbolsSet.count(s) == 0 && einsumPunctuation.find(s) == std::string::npos) {
      std::ostringstream o;
      o << "Character " << s << " is not a valid symbol.";
      throw ParseError(o.str());
    }
  }
}

std::string EinsumParser::findUniqueIndices(const std::string &subscripts) {

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
    dash_count += (subscripts[i] == '-');
    gt_count   += (subscripts[i] == '>');

    if(subscripts[i] == '-' && (i + 1) < (int)subscripts.size() && subscripts[i+1] != '>'){
      throw ParseError("Subscripts must contain '->' instead of '-'.");
    }
  }

  if(dash_count != gt_count || dash_count > 1){
    throw ParseError("Subscripts may only contain one '->'.");
  }

  return dash_count == 1;
}


std::string EinsumParser::replaceEllipse(std::string inp, std::string &newString){
  const std::string ellipse("...");
  size_t firstCharLoc = inp.find(ellipse);
  if(firstCharLoc == std::string::npos) {
    return inp;
  }

  std::string replaced(inp.begin(), inp.begin() + firstCharLoc);
  replaced += newString;
  replaced += std::string(inp.begin() + firstCharLoc + ellipse.size(), inp.end());
  return replaced;
}

std::vector<std::string> EinsumParser::splitSubscriptInput(std::string &input) {
  std::vector<std::string> splitInput;

  std::string betweenDelim;
  for(auto &sub : input) {
    if(sub == ',') {
      splitInput.push_back(betweenDelim);
      betweenDelim.clear();
    } else {
      betweenDelim += sub;
    }
  }
  splitInput.push_back(betweenDelim);
  return splitInput;
}

TensorBase& EinsumParser::getResultTensor() {
  return resultTensor;
}

void EinsumParser::buildResult(std::vector<std::string> subs){

  std::vector<std::string> inputSubs = splitSubscriptInput(subs[0]);
  std::string outputSubs = subs[1];

  std::map<char, IndexVar> subscriptToIndex;
  std::map<char, int> subscriptToShape;


  for(int currentInput = 0; currentInput < static_cast<int>(inputSubs.size()); ++currentInput){
    std::string &subscript = inputSubs[currentInput];
    if(static_cast<int>(subscript.size()) != tensors[currentInput].getOrder()) {
      std::ostringstream o;
      o << "Dimension mismatch for input " << currentInput << ".";
      throw ParseError(o.str());
    }

    const auto &tensorDims = tensors[currentInput].getDimensions();
    for(int i = 0; i < static_cast<int>(subscript.size()); ++i) {
      if(subscriptToIndex.count(i) == 0){
        subscriptToIndex.insert({subscript[i], IndexVar()});
      }

      // This is fine since we don't broadcast singleton dimensions meaning all indices must access the same var
      if(subscriptToShape.count(i) == 0) {
        subscriptToShape.insert({subscript[i], tensorDims[i]});
      } else if (subscriptToShape.at(subscript[i]) != tensorDims[i]) {
        throw ParseError("Dimensions mismatch.");
      }
    }
  }

  std::vector<IndexVar> vars;
  for(auto &sub : inputSubs[0]) {
    vars.push_back(subscriptToIndex.at(sub));
  }

  IndexExpr expr = tensors[0](vars);
  vars.clear();
  for(int i = 1; i < static_cast<int>(tensors.size()); ++i) {
    TensorBase& tensor = tensors[i];
    for(auto &sub : inputSubs[i]) {
      vars.push_back(subscriptToIndex.at(sub));
    }
    expr = expr * tensor(vars);
    vars.clear();
  }

  // Build output - first var list then tensor
  std::vector<int> outShape;
  for(auto &sub : outputSubs) {
    vars.push_back(subscriptToIndex.at(sub));
    outShape.push_back(subscriptToShape.at(sub));
  }

  if(format.getOrder() != 0 && format.getOrder() != static_cast<int>(outShape.size())) {
    std::ostringstream o;
    o << "Number of dimensions in format (" << format.getOrder() << ") does not match dimensions of output tensor("
                                                                 <<  outShape.size() << "),";
    throw ParseError(o.str());
  }

  if(format.getOrder() == 0 && format.getOrder() != static_cast<int>(outShape.size())) {
    format = Format(std::vector<ModeFormatPack>(outShape.size(), dense));
  }

  resultTensor = TensorBase(outType, outShape, format);
  resultTensor(vars) = expr;
}


void EinsumParser::parse() {

  bool hasOutput = exprHasOutput(subscripts);

  // Split tensor if output exists
  std::vector<std::string> inputTensorSubscripts;
  std::string outputSubscripts;
  if(hasOutput) {
    std::vector<std::string> temp = taco::util::split(subscripts, "->");
    outputSubscripts = temp.size() == 2? temp[1]: outputSubscripts;
    inputTensorSubscripts = splitSubscriptInput(temp[0]);
  }else {
    inputTensorSubscripts = splitSubscriptInput(subscripts);
  }

  if(inputTensorSubscripts.size() != tensors.size()) {
    throw ParseError("There needs to be one tensor for each input string.");
  }

  if(subscripts.find('.') != std::string::npos) {
    std::string usedSymbols, unusedSymbols;

    for(auto& elt : subscripts) {
      if(einsumPunctuation.find(elt) == std::string::npos) {
        usedSymbols += elt;
      }
    }

    for(auto& elt : einsumSymbols) {
      if (usedSymbols.find(elt) == std::string::npos) {
        unusedSymbols += elt;
      }
    }

    int longest = 0;

    for(size_t i = 0; i < inputTensorSubscripts.size(); ++i) {
      std::string subscript = inputTensorSubscripts[i];

      if(subscript.find('.') != std::string::npos) {
        if(std::count(subscript.begin(), subscript.end(), '.') != 3 || subscript.find("...") == std::string::npos) {
          throw ParseError("Invalid Ellipses.");
        }

        int ellipseCount = 0;
        if(tensors[i].getOrder() > 0){
          ellipseCount = std::max(tensors[i].getOrder(), 1);
          ellipseCount -= (subscript.size() - 3);
        }

        longest = std::max(longest, ellipseCount);

        if(ellipseCount < 0) {
          throw ParseError("Ellipses lengths do not match.");
        } else if (ellipseCount == 0) {
          std::string noEllipseSub;
          std::remove_copy(subscript.begin(), subscript.end(), std::back_inserter(noEllipseSub), '.');
          inputTensorSubscripts[i] = noEllipseSub;
        } else {
          std::string replace_ind(unusedSymbols.end() - ellipseCount, unusedSymbols.end());
          inputTensorSubscripts[i] = replaceEllipse(subscript, replace_ind);
        }
      }
    }

    subscripts = util::join(inputTensorSubscripts.begin(), inputTensorSubscripts.end(), ",");
    std::string outEllipse;
    if(longest > 0){
      outEllipse = std::string(unusedSymbols.end() - longest, unusedSymbols.end());
    }

    if(hasOutput) {
      subscripts += ("->" + replaceEllipse(outputSubscripts, outEllipse));
    } else {
      std::string sortedOutput = findUniqueIndices(subscripts);
      std::string normalInds;
      std::remove_copy_if(sortedOutput.begin(), sortedOutput.end(), std::back_inserter(normalInds),
                         [outEllipse](char chr){ return outEllipse.find(chr) != std::string::npos; });

      std::sort(normalInds.begin(), normalInds.end());
      subscripts += ("->" + outEllipse + normalInds);
    }
  }

  std::string inputSubs;
  if(subscripts.find("->") != std::string::npos){
    std::vector<std::string> temp = taco::util::split(subscripts, "->");
    outputSubscripts = temp.size() == 2? temp[1]: outputSubscripts;
    inputSubs = temp[0];
  } else {
    inputSubs = subscripts;
    outputSubscripts = findUniqueIndices(inputSubs);
    subscripts += ("->" + outputSubscripts);
  }


  for(auto &sub : outputSubscripts) {
    if(inputSubs.find(sub) == std::string::npos) {
      std::ostringstream o;
      o << "Output character " << sub << " did not appear in the input";
      throw ParseError(o.str());
    }
  }

  size_t num_inputs = splitSubscriptInput(inputSubs).size();
  if(num_inputs != tensors.size()) {
    throw ParseError("Number of einsum subscripts must be equal to the number of operands.");
  }

  buildResult({inputSubs, outputSubscripts});
}
}}
