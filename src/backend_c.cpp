#include "backend_c.h"
#include "ir_visitor.h"

using namespace std;

namespace tacit {
namespace internal {
class FindVars : public IRVisitor {
public:
  map<Expr, string, ExprCompare> var_map;
  
  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs)  {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      iassert(var) << "Inputs must be vars in codegen";
      iassert(var_map.count(var) == 0) << "Duplicate input found in codegen";
      var_map[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      iassert(var) << "Outputs must be vars in codegen";
      iassert(var_map.count(var) == 0) << "Duplicate output found in codegen";

      var_map[var] = var->name;
    }
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const Var *op) {
    if (var_map.count(op) == 0) {
      var_map[op] = CodeGen_C::gen_unique_name(op->name);
    }
  }
  
};

// Some helper functions

namespace {
// helper to translate from tacit type to C type
string to_c_type(ComponentType typ, bool is_ptr) {
  string ret;
  
  if (typ == typeOf<int>())
    ret = "int"; //TODO: should use a specific width here
  else if (typ == typeOf<float>())
    ret = "float";
  else if (typ == typeOf<double>())
    ret = "double";
  else
    iassert(false) << "Unknown type in codegen";
  
  if (is_ptr)
    ret += "*";
  
  return ret;
}

// helper to print declarations
string print_decls(map<Expr, string, ExprCompare> var_map,
                   vector<Expr> inputs, vector<Expr> outputs) {
  stringstream ret;
  for (auto varpair: var_map) {
    // make sure it's not an input or output
    if (find(inputs.begin(), inputs.end(), varpair.first) == inputs.end() &&
        find(outputs.begin(), outputs.end(), varpair.first) == outputs.end()) {
      auto var = varpair.first.as<Var>();
      ret << "  " << to_c_type(var->type, var->is_ptr);
      ret << " " << varpair.second << ";\n";
    }
  }
  return ret.str();
}

} // anonymous namespace

// initialize the counter for unique names to 0
int CodeGen_C::unique_name_counter = 0;

string CodeGen_C::gen_unique_name(string name) {
  // we add an underscore at the beginning in case this
  // is a keyword
  stringstream os;
  os << "_" << name << "_" << unique_name_counter++;
  return os.str();
}

CodeGen_C::CodeGen_C(std::ostream &dest) : IRPrinter(dest),
  func_block(true), out(dest) {  }
CodeGen_C::~CodeGen_C() { }

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_C::visit(const Var* op) {
  iassert(var_map.count(op) > 0) << "Var " << op->name << " not found in var_map";
  out << var_map[op];
}

void CodeGen_C::visit(const For* op) { IRPrinter::visit(op); }

void CodeGen_C::visit(const Block* op) {
  bool output_return = func_block;
  func_block = false;
  
  do_indent();
  out << "{\n";
  indent++;
  
  // if we're the first block in the function, we
  // need to print variable declarations
  out << func_decls;
  
  for (auto s: op->contents) {
    s.accept(this);
    out << ";\n";
  }
    
  if (output_return) {
    do_indent();
    out << "return 0;\n";
  }
  
  indent--;
  do_indent();
  out << "}\n";
}

void CodeGen_C::compile(const Function* func) {
  // find all the vars that are not inputs or outputs and declare them
  FindVars var_finder(func->inputs, func->outputs);
  func->body.accept(&var_finder);
  var_map = var_finder.var_map;

  func_decls = print_decls(var_map, func->inputs, func->outputs);

//  for (auto v : var_finder.var_map) {
//    cout << v.first << ": " << v.second << "\n";
//  }
  
  // output function declaration
  out << "int " << func->name << "(";
  for (size_t i=0; i<func->inputs.size(); i++) {
    auto var = func->inputs[i].as<Var>();
    iassert(var) << "inputs must be vars in codegen";
    
    out << to_c_type(var->type, var->is_ptr);
    out << " " << var->name;
    if (i != func->inputs.size()-1) {
      out << ", ";
    }
  }
  
  for (auto output: func->outputs) {
    auto var = output.as<Var>();
    iassert(var) << "outputs must be vars in codegen";
    
    out << ", ";
    out << to_c_type(var->type, var->is_ptr);
    out << " " << var->name;
  }
  out << ") ";
  
  // output body
  func->body.accept(this);
  
  // clear temporary stuff
  func_block = false;
  func_decls = "";

}




} // namespace internal
} // namespace tac
