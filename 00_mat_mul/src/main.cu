#include "TextTable.h"
#include <iostream>
#include <sys/time.h>

#define R_MIN 0
#define R_MAX 100

#define BLOCK_SIZE 32

struct FloatMatrix
{
    float **contents; // [y][x]
    int size;
};
struct Float1DMatrix
{
    float *contents; // [y * size + x]
    int size;
};

void dump_matrix(FloatMatrix *matrix)
{
    for (int i = 0; i < matrix->size; i++)
    {
        for (int j = 0; j < matrix->size; j++)
        {
            printf("%2.2lf\t ", matrix->contents[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

FloatMatrix *generate_matrix(int size)
{
    FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
    matrix->size = size;
    matrix->contents = (float **)malloc(size * sizeof(float *));
    for (int i = 0; i < size; i++)
        matrix->contents[i] = (float *)malloc(size * sizeof(float));
    return matrix;
}

void free_matrix(FloatMatrix *matrix)
{
    for (int i = 0; i < matrix->size; i++)
    {
        free((void *)matrix->contents[i]);
    }
    free((void *)matrix->contents);
    free((void *)matrix);
}

float random_float()
{
    return (R_MAX - R_MIN) * ((((float)rand()) / (float)RAND_MAX)) + R_MIN;
}

void rand_matrix(FloatMatrix *matrix)
{
    for (int i = 0; i < matrix->size; i++)
        for (int j = 0; j < matrix->size; j++)
            matrix->contents[i][j] = random_float();
}
Float1DMatrix *init_1dm_mat(int size)
{
    Float1DMatrix *f1dm = (Float1DMatrix *)malloc(sizeof(Float1DMatrix));
    f1dm->contents = (float *)malloc(size * size * sizeof(float));
    f1dm->size = size;
    return f1dm;
}
void free_1dm_mat(Float1DMatrix *f1dm)
{
    free((void *)f1dm->contents);
    free((void *)f1dm);
}
Float1DMatrix *convert_mat_to_1d(FloatMatrix *matrix)
{
    Float1DMatrix *f1dm = (Float1DMatrix *)malloc(sizeof(Float1DMatrix));
    f1dm->contents = (float *)malloc(matrix->size * matrix->size * sizeof(float));
    f1dm->size = matrix->size;
    for (int i = 0; i < matrix->size; i++)
        for (int j = 0; j < matrix->size; j++)
            f1dm->contents[i * matrix->size + j] = matrix->contents[i][j];
    return f1dm;
}
FloatMatrix *convert_1d_to_mat(Float1DMatrix *f1dm)
{
    FloatMatrix *mat = generate_matrix(f1dm->size);
    for (int i = 0; i < mat->size; i++)
        for (int j = 0; j < mat->size; j++)
            mat->contents[i][j] = f1dm->contents[i * mat->size + j];
    return mat;
}

FloatMatrix *native_c_matmul(FloatMatrix *m1, FloatMatrix *m2)
{
    FloatMatrix *result = generate_matrix(m1->size);
    for (int i = 0; i < result->size; i++)
        for (int j = 0; j < result->size; j++)
            for (int k = 0; k < result->size; k++)
                result->contents[i][j] += m1->contents[i][k] * m2->contents[k][j];
    return result;
}

void matmul_cpu(float *A, float *B, float *C, int m)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < m; l++)
            {
                sum += A[i * m + l] * B[l * m + j];
            }
            C[i * m + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < m)
    {
        float sum = 0.0f;
        for (int l = 0; l < m; l++)
        {
            sum += A[row * m + l] * B[l * m + col];
        }
        C[row * m + col] = sum;
    }
}

bool equals(FloatMatrix *matrix, float *result)
{
    for (int i = 0; i < matrix->size; i++)
        for (int j = 0; j < matrix->size; j++)
            if (abs(result[i * matrix->size + j] - matrix->contents[i][j]) > 0.000001)
                return false;
    return true;
}

int main()
{
    TextTable t('-', '|', '+');

    t.add("Matrix Size");
    t.add("Native C");
    t.add("CUDA CPU");
    t.add("CUDA GPU");
    t.add("CPU/GPU");
    t.endOfRow();

    clock_t start, end;
    for (int size = 100; size <= 2000; size += 100)
    {
        std::string fmt_matrix_size = std::to_string(size) + "x" + std::to_string(size);
        t.add(fmt_matrix_size);

        FloatMatrix *mat1 = generate_matrix(size);
        FloatMatrix *mat2 = generate_matrix(size);
        rand_matrix(mat1);
        rand_matrix(mat2);

        start = clock();
        FloatMatrix *mat3 = native_c_matmul(mat1, mat2);
        end = clock();
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        t.add(std::to_string(time_taken) + " sec");

        Float1DMatrix *f1dm1 = convert_mat_to_1d(mat1);
        Float1DMatrix *f1dm2 = convert_mat_to_1d(mat2);
        Float1DMatrix *f1dm3 = convert_mat_to_1d(mat3);

        float *result = (float *)malloc(f1dm1->size * f1dm2->size * sizeof(float));

        start = clock();
        matmul_cpu(f1dm1->contents, f1dm2->contents, result, size);
        end = clock();
        time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        t.add(std::to_string(time_taken) + " sec");

        if (!equals(mat3, result))
        {
            printf("Data is not equal\n");
        }
        else
        {
            printf("Data is equal\n");
        }

        float *d_A, *d_B, *d_C;
        int f1msize = mat1->size * mat1->size * sizeof(float);
        cudaMalloc(&d_A, f1msize);
        cudaMalloc(&d_B, f1msize);
        cudaMalloc(&d_C, f1msize);
        cudaMemcpy(d_A, f1dm1->contents, f1msize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, f1dm2->contents, f1msize, cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim(32, 32);

        start = clock();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
        cudaDeviceSynchronize();
        end = clock();

        double time_taken_1 = double(end - start) / double(CLOCKS_PER_SEC);
        t.add(std::to_string(time_taken_1));
        t.add(std::to_string(time_taken / time_taken_1));

        cudaMemcpy(result, d_C, f1msize, cudaMemcpyDeviceToHost);
        if (!equals(mat3, result))
        {
            printf("Data is not equal\n");
        }
        else
        {
            printf("Data is equal\n");
        }

        free_1dm_mat(f1dm1);
        free_1dm_mat(f1dm2);
        free_1dm_mat(f1dm3);

        free_matrix(mat1);
        free_matrix(mat2);
        free_matrix(mat3);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        t.endOfRow();
    }

    std::cout << t;
    return 0;
}
