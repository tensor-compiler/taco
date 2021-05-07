The Tensor Algebra Compiler (taco) is a C++ library that computes
tensor algebra expressions on sparse and dense tensors.  It uses novel
compiler techniques to get performance competitive with hand-optimized
kernels in widely used libraries for both sparse tensor algebra and
sparse linear algebra.

You can use taco as a C++ library that lets you load tensors, read
tensors from files, and compute tensor expressions.  You can also use
taco as a code generator that generates C functions that compute
tensor expressions.

Learn more about taco at
[tensor-compiler.org](https://tensor-compiler.org), in the paper
[The Tensor Algebra Compiler](http://tensor-compiler.org/kjolstad-oopsla17-tensor-compiler.pdf),
or in [this talk](https://youtu.be/Kffbzf9etLE).  To learn more about
where taco is going in the near-term, see the technical reports on
[optimization](https://arxiv.org/abs/1802.10574) and
[formats](https://arxiv.org/abs/1804.10112).

You can also subscribe to the
[taco-announcements](https://lists.csail.mit.edu/mailman/listinfo/taco-announcements)
email list where we post announcements, RFCs, and notifications of API
changes, or the [taco-discuss](https://lists.csail.mit.edu/mailman/listinfo/taco-discuss)
email list for open discussions and questions.

TL;DR build taco using CMake. Run `make test`.


# Build and test
![Build and Test](https://github.com/RSenApps/taco/workflows/Build%20and%20Test/badge.svg?branch=master)

Build taco using CMake 2.8.12 or greater:

    cd <taco-directory>
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8

## Building Python API
To build taco with the Python API (pytaco), add `-DPYTHON=ON` to the cmake line above. For example:

    cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON ..

You will then need to add the pytaco module to PYTHONPATH:

    export PYTHONPATH=<taco-directory>/build/lib:$PYTHONPATH

pytaco requires NumPy and SciPy to be installed.

## Building for OpenMP
To build taco with support for parallel execution (using OpenMP), add `-DOPENMP=ON` to the cmake line above. For example:

    cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON ..

## Building for CUDA
To build taco for NVIDIA CUDA, add `-DCUDA=ON` to the cmake line above. For example:

    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA=ON ..

Please also make sure that you have CUDA installed properly and that the following environment variables are set correctly:

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

If you do not have CUDA installed, you can still use the taco cli to generate CUDA code with the -cuda flag.

## Building for LLVM Backend
To build taco for LLVM, add `-DLLVM=ON` to the cmake line above. For example:

    cmake -DCMAKE_BUILD_TYPE=Release -DLLVM=ON ..

Make sure you have LLVM properly installed. 

## Running tests
To run all tests:

    cd <taco-directory>/build
    make test

Tests can be run in parallel by setting `CTEST_PARALLEL_LEVEL=<n>` in the environment (which runs `<n>` tests in parallel).

To run the C++ test suite individually:

    cd <taco-directory>
    ./build/bin/taco-test

To run the Python test suite individually:

    cd <taco-directory>
    python3 build/python_bindings/unit_tests.py


# Library example

The following sparse tensor-times-vector multiplication example in C++
shows how to use the taco library.

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
B.insert({1,2,1}, 3.0);
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

If you just need to compute a single tensor kernel you can use the
[taco online tool](http://www.tensor-compiler.org/online) to generate
a custom C library.  You can also use the taco command-line tool to
the same effect:

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

For more information, see our paper on the taco tools
[taco: A Tool to Generate Tensor Algebra Kernels](http://tensor-compiler.org/kjolstad-ase17-tools.pdf).
