The Tensor Algebra Compiler (tacit) compiles linear and tensor algebra expressions to code that operates on sparse and dense tensor formats. Tensor formats are repersented by multi-level trees and include common matrix formats such as CSR, CSC, ELL.

TL;DR build tacit using CMake. Run `tacit-tests` in the `bin` directory.

# Build and test the Tensor Algebra Compiler
Build tacit using CMake 2.8.3 or greater:

```
cd <tacit-directory>
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

Run the test suite:
```
cd <tacit-directory>
./build/bin/tacit-test
```
