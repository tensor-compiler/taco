# TACO testing

TACO has APIs and UIs in multiple languages and environments.
As such, there are currently 3 test suites for TACO:

1. C++ tests (googletest)
1. Python tests (python unittest)
1. Command line tests (BATS)

## invoking tests

All 3 of these test suites are run when you run `make test`.

### code coverage analysis

These 3 test suites are also run during code coverage analysis.  More details
of that can be found in the top level README.

## test suite details

### C++ (googletest)

The TACO C++ API is tested using the `googletest` testing framework.  The
tests are implemented in `.cpp` files contained within the `test/` folder.

The tests are linked into an executable called `taco-test`.  Individual tests
can be listed using the `--gtest_list_tests` parameter, or run individually using
the `--gtest_filter=<pattern>` parameter.  For example:

```sh
$ pwd
.../build
$ bin/taco-test --gtest_filter=scheduling.forallReplace
Note: Google Test filter = scheduling.forallReplace
[==========] Running 1 test from 1 test case.
[----------] Global test environment set-up.
[----------] 1 test from scheduling
[ RUN      ] scheduling.forallReplace
[       OK ] scheduling.forallReplace (0 ms)
[----------] 1 test from scheduling (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test case ran. (0 ms total)
[  PASSED  ] 1 test.
```

A copy of the `googletest` testing framework is bundled with TACO, in the
`test/gtest/` folder.

[Here](https://google.github.io/googletest/primer.html#simple-tests) is a starting guide to
writing test cases in googletest.

### Python (unittest)

The TACO Python API is tested using the python `unittest` module.  The tests
are implemented as subclasses of `unittest.TestCase` declared within the
`python_bindings/unit_tests.py` script.  A modified version of this script
is written into the `build/python_bindings` folder during TACO compilation,
and can be invoked directly from within that folder.  This modified version
hard-codes the build folder path, to ensure that the tests run against the
TACO library in the build folder, and not a system-installed TACO.

The tests can be invoked by going into the `build/python_bindings` folder
and running the `unit_tests.py` script.  Individual test cases can be run
using the `-k` parameter.  For example:

```sh
$ pwd
.../build/python_bindings
$ python3 unit_tests.py -k TestTensorCreation.test_tensor_from_numpy
test_tensor_from_numpy (__main__.TestTensorCreation) ... ok

----------------------------------------------------------------------
Ran 1 test in 1.023s

OK
```

[Here](https://docs.python.org/3/library/unittest.html#basic-example) is a
starting guide to writing Python test cases in unittest.

### Command line (BATS)

The TACO command line tool, `bin/taco`, is tested using the bash `bats-core`
testing framework.  The tests are implemented as `.bats` files contained
within the `test/` folder.

The test suite needs the `CMAKE_BUILD_DIR` environment variable to be set, so
it knows where to look for the `bin/taco` executable.  It should be set to the
folder you ran `cmake` in to configure and build TACO.  This variable is set
automatically when you run `make test`, but if you want to run tests by hand,
you will need to set it yourself.

Individual test cases can be run by running `bats` directly with the `-f`
parameter.  For example:

```sh
$ pwd
.../src
$ CMAKE_BUILD_DIR=../build test/bats/bin/bats -f layout test/
 ✓ test -f (tensor layout directives)

1 test, 0 failures

```

A copy of the `bats-core` testing framework is bundled with TACO as a
submodule, in the `test/bats/` folder.

[Here](https://bats-core.readthedocs.io/en/latest/writing-tests.html) is a
starting guide to writing test cases in bats.
