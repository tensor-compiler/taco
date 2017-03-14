The Tensor Algebra Compiler (taco) is a C++ library that compiles linear and tensor algebra expressions to code that operates on sparse and dense tensor formats. Tensor formats are repersented by multi-level trees and include common matrix formats such as CSR (Compressed Sparse Rows), CSC (Compressed Sparse Columns), DCSR (Doubly CSR), BCSR (Blocked CSR) and tensor formats such as CSF (Compressed Sparse Fibers).

TL;DR build taco using cmake. Run `taco-tests` in the `bin` directory.

# Build and test the Tensor Algebra Compiler
Build taco using CMake 2.8.3 or greater:

```
cd <taco-directory>
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

Run the test suite:
```
cd <taco-directory>
./build/bin/taco-test
```


# Tensor Algebra Compiler command line tool

The Tensor Algebra Compiler `taco` command line tool wraps the C++ library and let's you compiler tensor expressions to C99 code. It is intended mainly as a tensor library generator and development tool.

```
cd <taco-directory>
./build/bin/taco
Usage: taco [options] <index expression>

Examples:
  taco "a(i) = b(i) + c(i)"                            # Dense vector add
  taco "a(i) = b(i) + c(i)" -f=b:s -f=c:s -f=a:s       # Sparse vector add
  taco "a(i) = B(i,j) * c(j)" -f=B:ds                  # SpMV
  taco "A(i,l) = B(i,j,k) * C(j,l) * D(k,l)" -f=B:sss  # MTTKRP

Options:
  -f=<format>  Specify the format of a tensor in the expression. Formats are
               specified per dimension using d (dense) and s (sparse). All
               formats default to dense. Examples: A:ds, b:d and D:sss.

  -c           Print compute IR (default).

  -a           Print assembly IR.

  -nocolor     Print without colors.

Options planned for the future:
  -g           Generate random data for a given tensor. (e.g. B).

  -i           Initialize a tensor from an input file (e.g. B:"myfile.txt").
               If all the tensors have been initialized then the expression is
               evaluated.

  -o           Write the result of evaluating the expression to the given file

  -t           Time compilation, assembly and computation.
```
