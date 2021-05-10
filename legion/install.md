First, install Legion with CMake.

```
cd legion/
mkdir build
cd build
# Builds legion in debug mode.
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11"
# Builds legion in release mode.
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release
```

Then, use `find_package(Legion)` to use Legion in example codes.
