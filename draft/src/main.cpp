#include <chrono>
#include <iostream>
#include <cstring>
#include <string>

#include <vector_types.h>

__global__ void VecAdd(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
}