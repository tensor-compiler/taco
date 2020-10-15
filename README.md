# Reproducing OOPSLA 2020 Results
This branch is a snapshot of the code needed to generate the evaluated kernels and figures in the [OOPSLA 2020 paper](https://doi.org/10.1145/3428226). However, we highly recommend using the [master branch of taco](https://github.com/tensor-compiler/taco) or the [interactive web demo](http://tensor-compiler.org/codegen.html) if possible.

TACO is not generally able to generate parallel code for expressions where the output is stored in a format that does not support random inserts (such as CSR). This branch contains modifications to allow for generating parallel code in special cases where the output has the same sparsity pattern as one of the inputs. This is needed to evaluate kernels such as SDDMM (sampled dense-dense matrix multiplication) and TTV (tensor-times-vector) where the output is a CSR. However, these modifications break taco's ability to be used as a C++ library or generate correct assembly code (this causes for many of taco's test cases to fail). This branch can still be used to generate kernels using the instructions below, but we recommend using the master branch to generate kernels with a dense output.

The schedules used in the paper can be found in [test/test-scheduling-eval.cpp](test/tests-scheduling-eval.cpp).

To generate the kernels used in the paper:

    cd <taco-directory>
    git status # ensure you are on correct branch / commit (not master)
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    bin/taco-test --gtest_filter="generate_evaluation_files.*"
    
This will generate kernels in build/eval_prepared_cpu and build/eval_prepared_gpu. To add a kernel or modify a schedule, modify the [generate_evaluation_files test case](https://github.com/RSenApps/taco/blob/ea50ecfb9fa5fc0c8c9a435e5994f51a81ead427/test/tests-scheduling-eval.cpp#L1148).

It is also possible to use the command line tool, though there is no functionality in this branch to specify schedules via the command line. However, this is available in the master branch.

    bin/taco "y(i) = A(i,j) * x(j)" -f=A:ds -write-source=spmv.c
    vim spmv.c

Please contact [Ryan Senanayake (@RSenApps)](https://github.com/rsenapps/) for further questions on reproducing the OOPSLA 2020 results. We have a evaluation suite to generate the graphs used in the paper as well as a virtual machine with dependencies already set up.

# TACO Overview

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

TL;DR build taco using cmake. Run `taco-test` in the `bin` directory.


# Build and test 
![Build and Test](https://github.com/RSenApps/taco/workflows/Build%20and%20Test/badge.svg?branch=master)

Build taco using CMake 2.8.3 or greater:

    cd <taco-directory>
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
  
To build taco with support for parallel execution (using OpenMP), use the following cmake line with the instructions above:

    cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON ..

To build taco for NVIDIA CUDA, use the following cmake line with the instructions above:

    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA=ON ..

Please also make sure that you have CUDA installed properly and that the following environment variables are set correctly:
    
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
    
If you do not have CUDA installed, you can still use the taco cli to generate CUDA code with the -cuda flag

Run the test suite:

    cd <taco-directory>
    ./build/bin/taco-test


# Library Example

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
