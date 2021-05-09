First, install Legion with CMake.

```
cd legion/
mkdir build
cd build
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11"
```

Then, use `find_package(Legion)` to use Legion in example codes.
