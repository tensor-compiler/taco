#ifndef TACO_MODULE_H
#define TACO_MODULE_H

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <random>
#include <sstream>

#include "taco/target.h"
#include "taco/ir/ir.h"

namespace taco {
namespace ir {

class Module {
public:
    Module(Target target = getTargetFromEnvironment());

    std::string compile();
    void compileToSource(std::string path, std::string prefix);
    void compileToStaticLibrary(std::string path, std::string prefix);
    void addFunction(Stmt func);
    std::string getSource();
    void* getFuncPtr(std::string name);
    int callFuncPackedRaw(std::string name, std::vector<void*> args);
    int callFuncPacked(std::string name, std::vector<void*> args);

    void setSource(std::string source);

private:
    std::stringstream source;
    std::stringstream header;
    std::string libname;
    std::string tmpdir;
    void* lib_handle;
    std::vector<Stmt> funcs;
    bool moduleFromUserSource;
    Target target;

    static std::string chars;
    static std::default_random_engine gen;
    static std::uniform_int_distribution<int> randint;

    void setJITLibname();
    void setJITTmpdir();
};

} // namespace ir
} // namespace taco

#endif
