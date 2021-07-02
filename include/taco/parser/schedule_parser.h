#ifndef TACO_SCHEDULE_PARSER_H
#define TACO_SCHEDULE_PARSER_H

#include <string>
#include <vector>

namespace taco {
namespace parser {

// parse a string of the form: "reorder(i,j),precompute(D(i,j)*E(j,k),j,j_pre)"
// into string vectors of the form:
// [ [ "reorder", "i", "j" ], [ "precompute", "D(i,j)*E(j,k)", "j", "j_pre" ] ]
std::vector<std::vector<std::string>> ScheduleParser(const std::string);

std::vector<std::string> varListParser(const std::string);

// serialize the result of a parse (for debugging)
std::string serializeParsedSchedule(std::vector<std::vector<std::string>>);

}}

#endif //TACO_EINSUM_PARSER_H
