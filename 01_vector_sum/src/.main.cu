#include "TextTable.h"
#include <iostream>
#include <cuda_runtime.h>

#define R_MIN 0
#define R_MAX 100

#define MIN_SIZE 10
#define MAX_SIZE 1000000
#define SIZE_STEP 10

// Vector related
float *init_vector(int size)
{
    float *vec = (float *)malloc(size * sizeof(float));
    return vec;
}
float random_float()
{
    return (R_MAX - R_MIN) * ((((float)rand()) / (float)RAND_MAX)) + R_MIN;
}
void rand_fill(float *vec, int size)
{
    for (int i = 0; i < size; i++)
        vec[i] = random_float();
}
void dump_vec(float *vec, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
        printf(" %lf ", vec[i]);
    printf("]");
}
void free_vec(float *vec)
{
    free((void *)vec);
}

// Utils

bool equals(float a, float b)
{
    if (a == b)
        return true;
    return false;
}

void measure_time(
    float (*func)(float *, int),
    float *vec,
    int size,
    double *exec_time,
    float *result)
{
    clock_t start, end;
    start = clock();
    *result = func(vec, size);
    end = clock();
    *exec_time = double(end - start) / double(CLOCKS_PER_SEC);
}

// CPU
float naive_cpu_sum(float *vec, int size)
{
    float result = 0.0f;
    for (int i = 0; i < size; i++)
        result += vec[i];
    return result;
}

__global__ void naive_gpu_sum_kernel(float *A, float *C, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        C[i] = A[i] + 0.0f;
    }
}

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

float naive_gpu_sum(float *vec, int size)
{
    float *d_A = NULL;
    float *d_C = NULL;
    float *h_C = (float *)malloc(size * sizeof(float));

    cudaMalloc((void **)&d_A, size * sizeof(float));
    cudaMalloc((void **)&d_C, size * sizeof(float));
    cudaMemcpy(d_A, vec, size * sizeof(float), cudaMemcpyHostToDevice);

    // naive_gpu_sum_kernel<<<1, size>>>(d_A, d_C, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((size * sizeof(float)) + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A, d_C, size);
    // cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    dump_vec(h_C, size);

    return h_C[0];
}

int main()
{
    TextTable t('-', '|', '+');
    t.add("Vec Size");
    t.add("Naive");
    t.endOfRow();
    double exec_time;
    float result;
    for (int size = MIN_SIZE; size <= MAX_SIZE; size *= SIZE_STEP)
    {
        t.add(std::to_string(size));
        result = 0.0f;
        exec_time = 0.0;
        float *vec = init_vector(size);
        rand_fill(vec, size);

        measure_time(&naive_cpu_sum, vec, size, &exec_time, &result);
        t.add(std::to_string(exec_time) + " sec");
        float expected = result;

        measure_time(&naive_gpu_sum, vec, size, &exec_time, &result);
        printf("Result %d\n", result);
        if (!equals(expected, result))
            return -1;

        free_vec(vec);
        t.endOfRow();
        break;
    }
    std::cout << t;
    return 0;
}
