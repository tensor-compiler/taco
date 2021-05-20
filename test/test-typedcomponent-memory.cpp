#include <unistd.h>
#include <stdio.h>
#include <sys/mman.h>

#include "test.h"

#include "taco/tensor.h"

using namespace taco;

TEST(makecsr, access_past_pos) {
    int pagesize = getpagesize();
    int matrixsize = pagesize / sizeof(int) - 1;
    // carefully arrange pos[] array memory so that pos[1024] will segfault
    int *fence = (int*)mmap(NULL , pagesize*2, PROT_NONE             , MAP_PRIVATE | MAP_ANON            , -1, 0);
    int *pos   = (int*)mmap(fence, pagesize  , PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    int *crd = new int[matrixsize];
    double *value = new double[matrixsize];
    ASSERT_EQ(pos, fence) << "Unable to create guard zone for test";
    // 1023x1023 CSR identity matrix
    pos[matrixsize] = matrixsize;
    for(int i = 0; i < matrixsize; i++) {
        pos[i] = i;
        crd[i] = i;
        value[i] = 1.0;
    }
    {
        taco::Tensor<double> A = makeCSR("A", {matrixsize, matrixsize}, pos, crd, value);
    }
    delete []crd;
    delete []value;
    munmap(pos, pagesize);
    munmap(fence, pagesize*2);
    printf("Success!\n");
}
