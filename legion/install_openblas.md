# Installation

Install OpenBLAS as follows:
```
USE_OPENMP=1 make -j
make -j PREFIX=install install
// NOTE THAT PREFIX MUST BE AN ABSOLUTE PATH, NOT A RELATIVE PATH
```

If running with multiple OMP processors, you must uncomment the line in Makefile.rule of `NUM_PARALLEL=2` when building OpenBLAS.
