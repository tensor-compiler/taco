The Tensor Algebra Compiler (taco) is a C++ library that computes tensor
algebra expressions on sparse and dense tensors.  It uses novel compiler
techniques to get performance competitive hand-optimized kernels in widely used
libraries for both tensor algebra and linear algebra.

TL;DR build taco using cmake. Run `taco-test` in the `bin` directory.

# Build and test
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

# Example
The following sparse tensor-times-vector multiplication example shows how to
use the taco library.
```C++
// Create formats
Format csr({Dense,Sparse});
Format csf({Sparse,Sparse,Sparse});
Format  sv({Sparse});

// Create tensors
Tensor<double> A({2,3},   csr);
Tensor<double> B({2,3,4}, csf);
Tensor<double> c({4},     sv);

// Insert data into B and c
B.insert({0,0,0}, 1.0);
B.insert({1,2,0}, 2.0);
B.insert({1,3,1}, 3.0);
c.insert({0}, 4.0);
c.insert({1}, 5.0);

// Pack inserted data as described by the formats
B.pack();
c.pack();

// Form a tensor-vector multiplication expression
IndexVar i, j, k;
A(i,j) = B(i,j,k) * c(k);

// Compile the expression
A.compile();

// Assemble A's indices and numerically compute the result
A.assemble();
A.compute();

std::cout << A << std::endl;
```

# Code generation tools
If you just need to compute a single tensor kernel you can use the [taco online
tool](http://www.tensor-compiler.org/online) to generate a custom C library.  You can
also use the taco command-line tool to the same effect:
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
  ...
```
