#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>

#include "backend_c.h"
#include "ir_visitor.h"

using namespace std;

namespace taco {
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
// helper to translate from taco type to C type
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

// helper to count # of slots for a format
int format_slots(Format format) {
  int i = 0;
  for (auto level : format.getLevels()) {
    if (level.getType() == LevelType::Dense)
      i += 1;
    else
      i += 2;
  }
  i += 1; // for the vals
  return i;
}

// helper to unpack inputs and outputs
// inputs are unpacked to a pointer
// outputs are unpacked to a pointer
// TODO: this will change for tensors
string print_unpack(vector<Expr> inputs, vector<Expr> outputs) {
  stringstream ret;
  int slot = 0;
  
  for (auto input: inputs) {
    auto var = input.as<Var>();
    if (!var->is_tensor) {
      auto tp = to_c_type(var->type, var->is_ptr);
      // if the input is not of non-pointer type, we should unpack it
      // here
      auto deref = var->is_ptr ? "" : "*";
      ret << "  " << tp << " " << var->name;
      ret << " = " << deref << "(" << tp << deref << ")inputPack["
        << slot++ << "];\n";
    } else {
      ret << "  void** " << var->name << " = &(inputPack[" << slot << "]);\n";
      slot += format_slots(var->format);
    }
    
  }
  
  for (auto output: outputs) {
    auto var = output.as<Var>();
    if (!var->is_tensor) {

      iassert(var->is_ptr) << "Function outputs must be pointers";

      auto tp = to_c_type(var->type, var->is_ptr);
      ret << "  " << tp << " " << var->name << " = (" << tp << ")inputPack["
        << slot++ << "];\n";
    } else {
      ret << "  void** " << var->name << " = &(inputPack[" << slot << "]);\n";
      slot += format_slots(var->format);
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


void CodeGen_C::compile(Stmt stmt) {
  stmt.accept(this);
}

void CodeGen_C::visit(const Function* func) {
  // find all the vars that are not inputs or outputs and declare them
  FindVars var_finder(func->inputs, func->outputs);
  func->body.accept(&var_finder);
  var_map = var_finder.var_map;

  func_decls = print_decls(var_map, func->inputs, func->outputs);

  //  for (auto v : var_finder.var_map) {
  //    cout << v.first << ": " << v.second << "\n";
  //  }

  // output function declaration
  out << "int " << func->name << "(void** inputPack) ";
//  for (size_t i=0; i<func->inputs.size(); i++) {
//    auto var = func->inputs[i].as<Var>();
//    iassert(var) << "inputs must be vars in codegen";
//
//    out << to_c_type(var->type, var->is_ptr);
//    out << " " << var->name;
//    if (i != func->inputs.size()-1) {
//      out << ", ";
//    }
//  }

//  for (auto output: func->outputs) {
//    auto var = output.as<Var>();
//    iassert(var) << "outputs must be vars in codegen";
//
//    out << ", ";
//    out << to_c_type(var->type, var->is_ptr);
//    out << " " << var->name;
//  }
//  out << ") ";
  
  do_indent();
  out << "{\n";

  // input/output unpack
  out << print_unpack(func->inputs, func->outputs);

  // output body
  func->body.accept(this);
  
  do_indent();
  out << "}\n";

  // clear temporary stuff
  func_block = true;
  func_decls = "";
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_C::visit(const Var* op) {
  iassert(var_map.count(op) > 0) << "Var " << op->name << " not found in var_map";
  out << var_map[op];
}

string gen_vectorize_pragma(int width) {
  stringstream ret;
  ret << "#pragma clang loop interleave(enable) ";
  if (!width)
    ret << "vectorize(enable)";
  else
    ret << "vectorize_width(" << width << ")";
  
  return ret.str();
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Parallel, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_C::visit(const For* op) {
  if (op->kind == LoopKind::Vectorized) {
    do_indent();
    out << gen_vectorize_pragma(op->vec_width);
    out << "\n";
  }

  do_indent();
  out << "for (";
  op->var.accept(this);
  out << "=";
  op->start.accept(this);
  out << "; ";
  op->var.accept(this);
  out << "<";
  op->end.accept(this);
  out << "; ";
  op->var.accept(this);
  out << "+=";
  op->increment.accept(this);
  out << ")\n";
  do_indent();
  out << "{\n";
  
  if (!(op->contents.as<Block>())) {
    indent++;
    do_indent();
  }
  op->contents.accept(this);
  
  if (!(op->contents.as<Block>())) {
    indent--;
  }
  do_indent();
  out << "}\n";
}

void CodeGen_C::visit(const While* op) {
  // it's not clear from documentation that clang will vectorize
  // while loops
  // however, we'll output the pragmas anyway
  if (op->kind == LoopKind::Vectorized) {
    do_indent();
    out << gen_vectorize_pragma(op->vec_width);
    out << "\n";
  }

  do_indent();
  stream << "while (";
  op->cond.accept(this);
  stream << ")\n";
  do_indent();
  stream << "{\n";
  if (!(op->contents.as<Block>())) {
    indent++;
    do_indent();
  }
  op->contents.accept(this);
  
  if (!(op->contents.as<Block>())) {
    indent--;
  }
  do_indent();
  stream << "}\n";
}


void CodeGen_C::visit(const Block* op) {
  bool output_return = func_block;
  func_block = false;
  
  indent++;
  
  // if we're the first block in the function, we
  // need to print variable declarations
  if (output_return) {
    out << func_decls;
  }
  
  for (auto s: op->contents) {
    s.accept(this);
    if (!s.as<IfThenElse>()
        && !s.as<For>()
        && !s.as<While>()) {
      out << "\n";
    }
  }
    
  if (output_return) {
    do_indent();
    out << "return 0;\n";
  }
  
  indent--;

}

void CodeGen_C::visit(const GetProperty* op) {
  auto tensor = op->tensor.as<Var>();
  int slot = -1;
  string tp;
  
  // if we want the vals, we take the last slot
  if (op->property == TensorProperty::Values) {
    slot = format_slots(tensor->format) - 1;
    tp = to_c_type(tensor->type, tensor->is_ptr);
  } else {
    // for all others, we have to use the level
    auto levels = tensor->format.getLevels();
    iassert(op->dim < (int)levels.size()) << "Trying to access a nonexistent dimension";
    
    for (int i=0; i<op->dim; i++) {
      if (levels[i].getType() == LevelType::Dense)
        slot += 1;
      else
        slot += 2;
    }
    
    // for this level, if the property is index, we add 1
    if (op->property == TensorProperty::Index)
      slot += 1;
    
    // for a Dense level, nnz is an int
    // for a Fixed level, ptr is an int
    // all others are int*
    if ((levels[op->dim].getType() == LevelType::Dense &&
        op->property == TensorProperty::NNZ)
        ||(levels[op->dim].getType() == LevelType::Fixed &&
        op->property == TensorProperty::Pointer)) {
      tp = "int";
    } else {
      tp = "int*";
    }
  }
  out << "(" << tp << ")(";
  op->tensor.accept(this);
  out << "[" << slot << "])";
}


////// Module

Module::Module(string source) : source(source) {
  // Include stdio.h for printf
  this->source = "#include <stdio.h>\n" + this->source;

  // use POSIX logic for finding a temp dir
  char const *tmp = getenv("TMPDIR");
  if (!tmp) {
    tmp = "/tmp/";
  }
  tmpdir = tmp;
  
  // set the library name to some random alphanum string
  set_libname();
}

void Module::set_libname() {
  string chars = "abcdefghijkmnpqrstuvwxyz0123456789";
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[rand() % chars.length()];
}

string Module::compile() {
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";
  
  string cmd = "cc -std=c99 -g -shared " +
    prefix + ".c " +
    "-o " + prefix + ".so";

  // open the output file & write out the source
  ofstream source_file;
  source_file.open(prefix+".c");
  source_file << source;
  source_file.close();
  
  // now compile it
//  cout << "Executing " << cmd << endl;
  int err = system(cmd.data());
  uassert(err == 0) << "Compilation command failed:\n" << cmd
    << "\nreturned " << err;

  // use dlsym() to open the compiled library
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);

  return fullpath;
}

void* Module::get_func(std::string name) {
  void* ret = dlsym(lib_handle, name.data());
  uassert(ret != nullptr) << "Function " << name << " not found in module " <<
    tmpdir << libname;
  return ret;
}

int Module::call_func_packed(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  fnptr_t func_ptr = (fnptr_t)get_func(name);
  return func_ptr(args);
}

} // namespace internal
} // namespace taco
