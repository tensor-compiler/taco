If you want to use it as a standalone app, 
	Point the cmake build system to taco like so:

    export TACO_INCLUDE_DIR=<path to taco src dir>
    export TACO_LIBRARY_DIR=<path to taco lib dir>

Build the sptv example like so:

    mkdir build
    cd build
    cmake ..
    make

Run the explicit sptv example like so:

    ./sptv
