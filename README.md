The Tensor Algebra Compiler (taco) compiles linear and tensor algebra expressions to code that operates on sparse and dense tensor formats. Tensor formats are repersented by multi-level trees and include common matrix formats such as CSR, CSC, DCSR, BCSR, CSF.

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
