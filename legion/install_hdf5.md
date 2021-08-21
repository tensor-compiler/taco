To install HDF5 as needed by Legion follow these steps. Again, this
should only be done if the system supported HDF5 is inadequate for
some reason.

Download HDF5 from `http://sapling.stanford.edu/~manolis/hdf/hdf5-1.10.1.tar.gz`
and unpack it into some directory.

Within the HDF5 folder, run these commands.
```
./configure --prefix <full path to install dir> --enable-thread-safe --disable-hl
make -j
make -j install
```