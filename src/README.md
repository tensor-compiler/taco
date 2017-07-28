taco consists of two intertwined processes: one that compiles index expressions
to executable code and one that invokes executable code on inputs to produce
outputs.  In addition there are API's to create tensors from data or from index
expressions.


# Compilation

The compilation process consists of three code tranformations between four
representations:

**index expression**  --*lower*-->  **ir**  --*codegen*-->  **C/llvm**  --*compile*-->  **binary code**
