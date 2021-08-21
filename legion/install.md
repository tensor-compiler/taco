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
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release -DLegion_NETWORKS=gasnetex -DCMAKE_INSTALL_PREFIX="/home/rohany/cmake-install" -DLegion_EMBED_GASNet=true -DGASNet_CONDUIT=$CONDUIT
# Build legion on a remote host with OpenMP.
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release -DLegion_NETWORKS=gasnetex -DCMAKE_INSTALL_PREFIX="$HOME/cmake-install" -DLegion_EMBED_GASNet=true -DGASNet_CONDUIT=ibv -DLegion_USE_OpenMP=true
# Build legion on a remote host with OpenMP and CUDA.
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release -DLegion_NETWORKS=gasnetex -DCMAKE_INSTALL_PREFIX="$HOME/cmake-install" -DLegion_EMBED_GASNet=true -DGASNet_CONDUIT=ibv -DLegion_USE_OpenMP=true -DLegion_USE_CUDA=true
# If HDF5 is present, build with it by adding the following:
cmake ../ -DCMAKE_CXX_FLAGS="--std=c++11" -DCMAKE_BUILD_TYPE=Release -DLegion_NETWORKS=gasnetex -DCMAKE_INSTALL_PREFIX="$HOME/cmake-install" -DLegion_EMBED_GASNet=true -DGASNet_CONDUIT=ibv -DLegion_USE_OpenMP=true -DLegion_USE_CUDA=true -DLegion_USE_HDF5=true
```

When interacting with sparse data structures, we require HDF5 for
file IO purposes. To build Legion with HDF5 support, add `-DLegion_USE_HDF5=true` to
the build command line. This works if you are going to use the system installation 
of HDF5. There are cases where this installation is not appropriate (such as if the
installation only has the parallel build). For these cases, you must follow the 
directions in `install_hdf5.md` to create a local installation of HDF5. Then,
use the same build command, but prepend it with `HDF5_ROOT=path/to/hdf5/install/dir`
to tell CMake to use the separate build of HDF5.

Then, use `find_package(Legion)` to use Legion in example codes.
