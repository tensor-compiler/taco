#include <string>
#include <vector>
#include <iostream>

#include "taco/parser/lexer.h"
#include "taco/parser/schedule_parser.h"
#include "taco/error.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;

namespace taco{
namespace parser{

/// Parses command line schedule directives (`-s <directive>`).
/// Example: "precompute(B(i,j),j,jpre),reorder(j,i)" is parsed as:
/// [ [ "precompute", "B(i,j)", "j", "jpre" ],
///   [ "reorder", "j", "i" ] ]
/// The first element of each inner vector is the function name.
/// Inner parens are preserved.  All whitespace is removed.
vector<vector<string>> ScheduleParser(const string argValue) {
    int parenthesesCnt;
    vector<vector<string>> parsed;
    vector<string> current_schedule;
    string current_element;
    parser::Lexer lexer(argValue);
    parser::Token tok;
    parenthesesCnt = 0;
    int curlyParenthesesCnt = 0;

    for(tok = lexer.getToken(); tok != parser::Token::eot; tok = lexer.getToken()) {
        switch(tok) {
        case parser::Token::lparen:
            if(parenthesesCnt == 0) {
                // The first opening paren separates the name of the scheduler directive from its first parameter
                current_schedule.push_back(current_element);
                current_element = "";
            }
            else {
                // pass inner parens through to the scheduler
                current_element += lexer.tokenString(tok);
            }
            parenthesesCnt++;
            break;
        case parser::Token::rparen:
            taco_uassert(parenthesesCnt > 0) << "mismatched parentheses" 
                                                "(too many right-parens, negative nesting level) "
                                                "in schedule expression '" << argValue << "'";
            if(parenthesesCnt > 1)
                current_element += lexer.tokenString(tok);
            parenthesesCnt--;
            break;
        case parser::Token::comma:
            if (curlyParenthesesCnt > 0) {
              // multiple indexes inside of a {} list; pass it through
              current_element += lexer.tokenString(tok);
            } else if(parenthesesCnt == 0) {
                // new schedule directive
                current_schedule.push_back(current_element);
                parsed.push_back(current_schedule);
                current_schedule.clear();
                current_element = "";
            } else if(parenthesesCnt == 1) {
                // new parameter to schedule directive
                current_schedule.push_back(current_element);
                current_element = "";
            } else {
                // probably multiple indexes inside of an IndexExpr; pass it through
                current_element += lexer.tokenString(tok);
                break;
            }
            break;
        case parser::Token::lcurly:
            // Keep track of curly brackets for list arguments
            current_element += lexer.tokenString(tok);
            curlyParenthesesCnt++;
            break;
        case parser::Token::rcurly:
            taco_uassert(curlyParenthesesCnt > 0) << "mismatched curly parentheses "
                                                     "(too many right-curly-parens, negative nesting level)"
                                                     " in schedule expression '" << argValue << "'";
            current_element += lexer.tokenString(tok);
            curlyParenthesesCnt--;
            break;
        // things where .getIdentifier() makes sense
        case parser::Token::identifier:
        case parser::Token::int_scalar:
        case parser::Token::uint_scalar:
        case parser::Token::float_scalar:
        case parser::Token::complex_scalar:
            current_element += lexer.getIdentifier();
            break;
        // .tokenstring() works for the remaining cases
        default:
            current_element += lexer.tokenString(tok);
            break;
        }
    }
    taco_uassert(parenthesesCnt == 0) << "imbalanced parentheses (too few right-parens) "
                                         "in schedule expression '" << argValue << "'";
    if(current_element.length() > 0)
        current_schedule.push_back(current_element);
    if(current_schedule.size() > 0)
        parsed.push_back(current_schedule);
    return parsed;
}

/// Parses command line lists for the scheduling directive 'precompute(expr, i_vars, iw_vars)'
/// The lists are used for i_vars and iw_vars
vector<string> varListParser(const string argValue) {
  vector<string> parsed;
  string current_element;
  parser::Lexer lexer(argValue);
  parser::Token tok;
  int curlyParenthesesCnt = 0;

  for(tok = lexer.getToken(); tok != parser::Token::eot; tok = lexer.getToken()) {
    switch(tok) {
      case parser::Token::comma:
        if (curlyParenthesesCnt > 0) {
          // multiple indexes inside of a {} list; pass it through
          parsed.push_back(current_element);
          current_element = "";
        } else {
          // probably multiple indexes inside of an IndexExpr; pass it through
          current_element += lexer.tokenString(tok);
          break;
        }
        break;
      case parser::Token::lcurly:
        // Keep track of curly brackets for list arguments
        current_element = "";
        curlyParenthesesCnt++;
        break;
      case parser::Token::rcurly:
        taco_uassert(curlyParenthesesCnt > 0) << "mismatched curly parentheses "
                                                 "(too many right-curly-parens, negative nesting level)"
                                                 " in schedule expression '" << argValue << "'";
        if (curlyParenthesesCnt == 1) {
            parsed.push_back(current_element);
            current_element = "";
        }
        curlyParenthesesCnt--;
        break;
      case parser::Token::lparen:
        // ignore parenthesis
        break;
      case parser::Token::rparen:
        // ignore parenthesis
        break;
        // things where .getIdentifier() makes sense
      case parser::Token::identifier:
      case parser::Token::int_scalar:
      case parser::Token::uint_scalar:
      case parser::Token::float_scalar:
      case parser::Token::complex_scalar:
        current_element += lexer.getIdentifier();
        break;
        // .tokenstring() works for the remaining cases
      default:
        current_element += lexer.tokenString(tok);
        break;
    }
  }
  taco_uassert(curlyParenthesesCnt == 0) << "imbalanced curly brackets (too few right-curly brackets) "
                                            "in schedule expression '" << argValue << "'";
  if(current_element.length() > 0)
    parsed.push_back(current_element);
  return parsed;
}



string serializeParsedSchedule(vector<vector<string>> parsed) {
    std::stringstream ss;
    ss << "[ ";
    for(vector<string> current_schedule : parsed) {
        ss << "[ ";
        for(string element : current_schedule) {
            ss << "'" << element << "', ";
        }
        ss << "], ";
    }
    ss << "]";
    return ss.str();
}
}}
