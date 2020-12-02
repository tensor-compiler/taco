// This file tests matrix-matrix multiply, permuting matrix formats for all 3 matrixes.
#include <stdio.h>
#include "test.h"
#include "taco/tensor.h"

using namespace std;
using namespace taco;

static int N = 5;

enum tensor_type {
    tensor_type_dense = 0,
    tensor_type_csr,
    tensor_type_csc,
    tensor_type_coo,
    _last_tensor_type
};

static map<int, const char*> tensor_type_names = {
    {tensor_type_dense, "Dense"},
    {tensor_type_csr  , "CSR"},
    {tensor_type_csc  , "CSC"},
    {tensor_type_coo  , "COO"},
};

static Tensor<double> create_2d_tensor_type(string name, int format) {
    switch(format) {
    case tensor_type_csr: {
            Tensor<double> rv(name, {N,N}, CSR);
            return rv;
        }
    case tensor_type_csc: {
            Tensor<double> rv(name, {N,N}, CSC);
            return rv;
        }
    case tensor_type_coo: {
            Tensor<double> rv(name, {N,N}, COO(2));
            return rv;
        }
    case tensor_type_dense: {
            Tensor<double> rv(name, {N,N}, Format({Dense,Dense}));
            return rv;
        }
    default: {
            throw;
        }
    }
}

bool try_sparse_output_permutation(int A_type, int B_type, int C_type, Tensor<double> &expected) {
    /* Sparse Matrix Multiplication */

    Tensor<double> A = create_2d_tensor_type("A", A_type);
    Tensor<double> B = create_2d_tensor_type("B", B_type);
    Tensor<double> C = create_2d_tensor_type("C", C_type);

    B(0,0) = 1.0;
    B(1,0) = 2.0;
    B(2,0) = 3.0;
    B(2,1) = 4.0;
    B(3,2) = 5.0;
    B.pack();

    C(0,0) = 1.0;
    C(1,0) = 2.0;
    C(2,0) = 3.0;
    C(2,1) = 4.0;
    C(3,2) = 5.0;
    C.pack();

    IndexVar i, j, k;
    A(i,j) = B(i,k) * C(k,j);

    A.evaluate();

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            if(abs(A(i,j) - expected(i,j)) > 0.000001)
                return false;

    return true;
}

TEST(format, DISABLED_mm_permute_formats) {
    printf("| Result | A =   | B *   | C     |\n");
    printf("|--------|-------|-------|-------|\n");
    bool failed = false;
    Tensor<double> expected = create_2d_tensor_type("expected", tensor_type_csr);
    expected(0, 0) = 1;
    expected(1, 0) = 2;
    expected(2, 0) = 11;
    expected(3, 0) = 15;
    expected(3, 1) = 20;
    expected.pack();
    for(int A_type = 0; A_type < _last_tensor_type; A_type++) {
        for(int B_type = 0; B_type < _last_tensor_type; B_type++) {
            for(int C_type = 0; C_type < _last_tensor_type; C_type++) {
                bool result;
                try {
                    result = try_sparse_output_permutation(A_type, B_type, C_type, expected);
                } catch (int e) {
                    result = false;
                }
                printf("| %-6s | %-5s | %-5s | %-5s |\n",
                    result == true ? "pass" : "FAIL",
                    tensor_type_names[A_type],
                    tensor_type_names[B_type],
                    tensor_type_names[C_type]);
                if(result == false)
                    failed = true;
            }
        }
    }
    if(failed == true)
        FAIL() << "at least one permutation of matrix formats failed";
}
