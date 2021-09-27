#ifndef TACO_LEGION_STRINGS_H
#define TACO_LEGION_STRINGS_H

#include <string>
#include <vector>

// Split a string into components based on delim.
std::vector<std::string> split(const std::string &str, const std::string &delim, bool keepDelim = false);

// Check if a string ends with another.
bool endsWith(std::string const &fullString, std::string const &ending);

#endif // TACO_LEGION_STRINGS_H