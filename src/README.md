taco consists of two intertwined processes: one that compiles index
expressions to executable code and one that invokes executable code on
inputs to produce outputs. The following layer diagram shows the main
modules and how they related. 

![Modules](https://github.com/tensor-compiler/taco/wiki/images/modules.png)

# Compilation

The compilation process consists of three code tranformations between four
representations:

**index expression**  --*lower*-->  **ir**  --*codegen*-->  **C/llvm**  --*compile*-->  **binary code**
