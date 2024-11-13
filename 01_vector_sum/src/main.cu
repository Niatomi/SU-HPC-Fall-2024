#include "TextTable.h"
#include <iostream>
#include <cuda_runtime.h>

#define R_MIN 0
#define R_MAX 100

#define MIN_SIZE 10
#define MAX_SIZE 1000000000
#define SIZE_STEP 10

#define THREADS_PER_BLOCK 512

// Vector
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
    printf("]\n");
}
void free_vec(float *vec)
{
    free((void *)vec);
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
    double result = 0.0f;
    for (int i = 0; i < size; i++)
        result += vec[i];
    return result;
}

// GPU
// Naive
// vvvvvv...
//  sum(s) = result
__global__ void naive_gpu_sum_kernel(float *A, float *C, int size)
{
    int i = threadIdx.x;
    if (i < size)
    {
        atomicAdd(&C[0], A[i]);
    }
}
float naive_gpu_sum(float *vec, int size)
{
    float *d_A = NULL;
    float *d_C = NULL;
    float *h_C = (float *)malloc(sizeof(float));

    cudaMalloc((void **)&d_A, size * sizeof(float));
    cudaMalloc((void **)&d_C, sizeof(float));
    cudaMemcpy(d_A, vec, size * sizeof(float), cudaMemcpyHostToDevice);

    naive_gpu_sum_kernel<<<1, size>>>(d_A, d_C, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
    return h_C[0];
}

// Half Split sum
// vvvvvvvvvv|vvvvvvvvvv
//
// vvvvvvvvvv|
// vvvvvvvvvv| +
// ----------|
// vvvvvvvvvv|
//     s = result
__global__ void s_kernel_worker(
    float *A,
    float *B,
    int size)
{
    int i = threadIdx.x;
    if (i < ((int)size / 2))
    {
        A[i] = B[i] + B[((int)size / 2) + i];
    }
}
__global__ void split_gpu_sum_kernel(float *A, float *C, int size)
{
    __syncthreads();

    if (threadIdx.x == 0)
    {
        s_kernel_worker<<<1, ((int)size / 2)>>>(A, A, size);
        __syncthreads();
        naive_gpu_sum_kernel<<<1, ((int)size / 2)>>>(A, C, ((int)size / 2));
    }
}

float split_gpu_sum(float *vec, int size)
{
    float *d_A = NULL;
    float *d_C = NULL;
    float *h_C = (float *)malloc(sizeof(float));

    cudaMalloc((void **)&d_A, size * sizeof(float));
    cudaMalloc((void **)&d_C, sizeof(float));
    cudaMemcpy(d_A, vec, size * sizeof(float), cudaMemcpyHostToDevice);

    split_gpu_sum_kernel<<<1, 1>>>(d_A, d_C, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_C);
    return h_C[0];
}

// Split By size / THREADS_PER_BLOCK
__global__ void shared_sum_kernels(float *result, float *vector, int size, int chunk_size)
{
    __shared__ float s[THREADS_PER_BLOCK];
    int id = threadIdx.x;
    int l_idx = chunk_size * id;
    int r_idx = chunk_size * id + chunk_size;
    if (r_idx > size)
        r_idx = size + 1;
    s[id] = 0.0f;
    if (l_idx < size && r_idx <= size)
    {
        for (int i = l_idx; i < r_idx; i++)
        {
            s[id] += vector[i];
        }
        result[id] = s[id];
    }
}
float shared_sum_host_count(float *vector, int size)
{
    float result = 0.0f;
    float *h_result = (float *)malloc(THREADS_PER_BLOCK * sizeof(float));
    float *d_result;
    float *d_vector;

    cudaMalloc((void **)&d_vector, size * sizeof(float));
    cudaMemcpy(d_vector, vector, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, size * sizeof(float));
    int csize = ceil((size / THREADS_PER_BLOCK) + 1);
    shared_sum_kernels<<<1, THREADS_PER_BLOCK>>>(d_result, d_vector, size, csize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < THREADS_PER_BLOCK; i++)
        result += h_result[i];

    cudaFree(d_result);
    cudaFree(d_vector);
    free(h_result);

    return result;
}

__global__ void shared_sum_kernels_atomic(float *result, float *vector, int size, int chunk_size)
{
    __shared__ float s[THREADS_PER_BLOCK];
    int id = threadIdx.x;
    int l_idx = chunk_size * id;
    int r_idx = chunk_size * id + chunk_size;
    if (r_idx > size)
        r_idx = size + 1;
    s[id] = 0.0f;
    if (l_idx < size && r_idx <= size)
    {
        for (int i = l_idx; i < r_idx; i++)
        {
            s[id] += vector[i];
        }
        atomicAdd(result, s[id]);
    }
}
float shared_sum_atomic_count(float *vector, int size)
{
    float result = 0.0f;
    float *h_result = (float *)malloc(THREADS_PER_BLOCK * sizeof(float));
    float *d_result;
    float *d_vector;

    cudaMalloc((void **)&d_vector, size * sizeof(float));
    cudaMemcpy(d_vector, vector, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, size * sizeof(float));
    int csize = ceil((size / THREADS_PER_BLOCK) + 1);
    shared_sum_kernels_atomic<<<1, THREADS_PER_BLOCK>>>(d_result, d_vector, size, csize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);
    result = h_result[0];
    cudaFree(d_result);
    cudaFree(d_vector);
    free(h_result);

    return result;
}

int main()
{
    TextTable t('-', '|', '+');
    t.add("Vec Size");
    t.add("Naive CPU");
    t.add("Naive GPU");
    t.add("Half Split GPU");
    t.add("(S/512) Split GPU (ДХ)");
    t.add("(S/512) Split GPU (ДД)");
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

        measure_time(&naive_gpu_sum, vec, size, &exec_time, &result);
        t.add(std::to_string(exec_time) + " sec");

        measure_time(&split_gpu_sum, vec, size, &exec_time, &result);
        t.add(std::to_string(exec_time) + " sec");

        measure_time(&shared_sum_host_count, vec, size, &exec_time, &result);
        t.add(std::to_string(exec_time) + " sec");

        measure_time(&shared_sum_atomic_count, vec, size, &exec_time, &result);
        t.add(std::to_string(exec_time) + " sec");

        free_vec(vec);
        t.endOfRow();
    }
    std::cout << t;
    return 0;
}
