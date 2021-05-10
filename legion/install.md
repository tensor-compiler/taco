First, install Legion with CMake.

```
cd legion/
mkdir build
cd build
# Builds legion in debug mode.
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11"
# Builds legion in release mode.
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release
# Builds legion on a remote host (with GASNet).
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release -DLegion_NETWORKS=gasnet1 -DCMAKE_INSTALL_PREFIX="/home/rohany/cmake-install" -DLegion_EMBED_GASNet=true -DGASNet_CONDUIT=$CONDUIT
```

Then, use `find_package(Legion)` to use Legion in example codes.
