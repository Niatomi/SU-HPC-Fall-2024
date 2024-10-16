#include "TextTable.h"
#include <iostream>
#include <sys/time.h>

#define R_MIN 0
#define R_MAX 100

#define ID_CPU 1
#define ID_GPU 2

#define M 100 // Number of rows in A and C
#define K 100 // Number of columns in A and rows in B
#define N 100 // Number of columns in B and C
#define BLOCK_SIZE 3200

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

float randomFloat()
{
    return (R_MAX - R_MIN) * ((((float)rand()) / (float)RAND_MAX)) + R_MIN;
}

void rand_matrix(FloatMatrix *matrix)
{
    for (int i = 0; i < matrix->size; i++)
        for (int j = 0; j < matrix->size; j++)
            matrix->contents[i][j] = randomFloat();
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
            float sum = 10.0f;
            for (int l = 0; l < m; l++)
            {
                sum += A[i * m + l] * B[l * m + j];
            }
            C[i * m + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 1.0f;
    C[0] = 255.0f;
    // if (row < m && col < n)
    // {
    //     float sum = -1.0f;
    //     for (int l = 0; l < k; l++)
    //     {
    //         sum += A[row * k + l] * B[l * n + col];
    //     }
    //     C[row * n + col] = sum;
    // }
}

int main()
{
    TextTable t('-', '|', '+');

    t.add("Matrix Size");
    t.add("Native C");
    t.add("CUDA CPU");
    // t.add("CUDA GPU");
    t.endOfRow();

    // for (int size = 3; size <= 2000; size += 100)
    // {
    int size = 3;
    std::string fmt_matrix_size = std::to_string(size) + "x" + std::to_string(size);
    t.add(fmt_matrix_size);

    FloatMatrix *mat1 = generate_matrix(size);
    FloatMatrix *mat2 = generate_matrix(size);
    rand_matrix(mat1);
    rand_matrix(mat2);

    clock_t start, end;
    start = clock();

    FloatMatrix *mat3 = native_c_matmul(mat1, mat2);
    dump_matrix(mat3);

    Float1DMatrix *f1dm1 = convert_mat_to_1d(mat1);
    Float1DMatrix *f1dm2 = convert_mat_to_1d(mat2);
    float *result = (float *)malloc(f1dm1->size * f1dm2->size * sizeof(float));
    matmul_cpu(f1dm1->contents, f1dm2->contents, result, size);
    cudaDeviceSynchronize();
    Float1DMatrix *f1dmr = init_1dm_mat(f1dm1->size);
    f1dmr->contents = result;

    FloatMatrix *fmdmr = convert_1d_to_mat(f1dmr);
    dump_matrix(fmdmr);

    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

    t.add(std::to_string(time_taken) + " sec");
    t.endOfRow();

    float *d_A, *d_B, *d_C;
    int f1msize = mat1->size * mat1->size * sizeof(float);
    cudaMalloc(&d_A, f1msize);
    cudaMalloc(&d_B, f1msize);
    cudaMalloc(&d_C, f1msize);
    f1dm1 = convert_mat_to_1d(mat1);
    f1dm2 = convert_mat_to_1d(mat2);
    f1dmr = init_1dm_mat(f1dm1->size);
    cudaMemcpy(d_A, f1dm1->contents, f1msize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, f1dm2->contents, f1msize, cudaMemcpyHostToDevice);

    // int numBlocks = (10000000 + mat1->size * mat1->size - 1) / mat1->size;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_gpu<<<10000, 1024>>>(d_A, d_B, d_C, size, size, size);
    int ret = cudaDeviceSynchronize();
    printf("%d\n", ret);
    float *c = (float *)malloc(f1msize);
    cudaMemcpy(c, d_C, f1msize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < mat1->size; i++)
    {
        printf("%2.2lf\t ", c[i]);
    }

    // float *c = (float *)malloc(size);

    // free_matrix(mat1);
    // free_matrix(mat2);

    // std::cout << t;
    return 0;
}
