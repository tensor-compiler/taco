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
            taco_uassert(parenthesesCnt > 0) << "mismatched parentheses (too many right-parens, negative nesting level) in schedule expression '" << argValue << "'";
            if(parenthesesCnt > 1)
                current_element += lexer.tokenString(tok);
            parenthesesCnt--;
            break;
        case parser::Token::comma:
            if(parenthesesCnt == 0) {
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
    taco_uassert(parenthesesCnt == 0) << "imbalanced parentheses (too few right-parens) in schedule expression '" << argValue << "'";
    if(current_element.length() > 0)
        current_schedule.push_back(current_element);
    if(current_schedule.size() > 0)
        parsed.push_back(current_schedule);
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
