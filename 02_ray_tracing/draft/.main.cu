#include <curand_kernel.h>
#include <stdio.h>

#include "cuda_utils.h"

// Implement a simple ray tracing algorithm without refraction rays using GPU.
//
// The generated scene should consist:
// - 5-10 spheres of different colors
// - 1 or 2 point-like light sources.
//
// Maximum depth of recursion is 5.
// All objects in the scene are not transparent.

#define ASPECT_RATIO 1.777778 // 16/9
#define PICTURES_AMMOUNT 10
#define MIN_WIDTH 800
#define MAX_WIDTH 1920
#define REFLECT_RECURESION_LIMIT 5

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = blockIdx.y;
    int j = blockIdx.x;
    if ((i >= max_y) || (j >= max_x))
        return;
    int pixel_index = i * max_x + j;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}

int main()
{
    clock_t start, stop;
    int image_width = MIN_WIDTH;
    int width_step = (MAX_WIDTH - MIN_WIDTH) / PICTURES_AMMOUNT;
    for (; image_width <= MAX_WIDTH; image_width += width_step)
    {
        start = clock();
        int image_height = ceil(image_width / ASPECT_RATIO);
        image_height = (image_height < 1) ? 1 : image_height;
        printf("Rendering %dx%d\n", image_width, image_height);

        curandState *d_rand_state;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_rand_state, image_width * image_width * sizeof(curandState)));
        dim3 blocks(image_width, image_height);
        dim3 threads(1);
        render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        stop = clock();
        printf("Render finished in %lf seconds\n", ((double)(stop - start)) / CLOCKS_PER_SEC);

        for (int j = 0; j < image_height; j++)
        {
            for (int i = 0; i < image_width; i++)
            {

                // auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
                // auto ray_direction = pixel_center - camera_center;
                // ray r(camera_center, ray_direction);
                // color pixel_color = ray_color(r, world);

                // write_pixel(img, i, j, pixel_color);
            }
        }

        return 0;
    }

    // // allocate random state
    // curandState *d_rand_state;
    // checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // // make our world of hitables & the camera
    // hitable **d_list;
    // checkCudaErrors(cudaMalloc((void **)&d_list, 4 * sizeof(hitable *)));
    // hitable **d_world;
    // checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    // camera **d_camera;
    // checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    // create_world<<<1, 1>>>(d_list, d_world, d_camera);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    // clock_t start, stop;
    // start = clock();
    // // Render our buffer
    // dim3 blocks(nx / tx + 1, ny / ty + 1);
    // dim3 threads(tx, ty);
    // render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // stop = clock();
    // double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    // std::cerr << "took " << timer_seconds << " seconds.\n";

    // // Output FB as Image
    // std::cout << "P3\n"
    //           << nx << " " << ny << "\n255\n";
    // for (int j = ny - 1; j >= 0; j--)
    // {
    //     for (int i = 0; i < nx; i++)
    //     {
    //         size_t pixel_index = j * nx + i;
    //         int ir = int(255.99 * fb[pixel_index].r());
    //         int ig = int(255.99 * fb[pixel_index].g());
    //         int ib = int(255.99 * fb[pixel_index].b());
    //         std::cout << ir << " " << ig << " " << ib << "\n";
    //     }
    // }

    // // clean up
    // checkCudaErrors(cudaDeviceSynchronize());
    // free_world<<<1, 1>>>(d_list, d_world, d_camera);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaFree(d_camera));
    // checkCudaErrors(cudaFree(d_world));
    // checkCudaErrors(cudaFree(d_list));
    // checkCudaErrors(cudaFree(d_rand_state));
    // checkCudaErrors(cudaFree(fb));

    // cudaDeviceReset();
}