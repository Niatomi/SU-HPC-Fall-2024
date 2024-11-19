#include "image.h"
#include "cuda_utils.h"

#include <stdio.h>
#include <vector>
#include <math.h>

// Implement a simple ray tracing algorithm without refraction rays using GPU.
//
// The generated scene should consist:
// - 5-10 spheres of different colors
// - 1 or 2 point-like light sources.
//
// Maximum depth of recursion is 5.
// All objects in the scene are not transparent.

/*
TraceRay(ray, depth)
{
    if(depth > maximal depth)
        return 0;

    find closest ray object/intersection;
    if(intersection exists)
    {
        for each light source in the scene
        {
            if(light source is visible)
            {
                illumination += light contribution;
            }
        }
        if(surface is reflective)
        {
            illumination += TraceRay(reflected ray, depth+1) ;
        }
        return illumination modulated according to the surface properties;
    }
    else return EnvironmentMap(ray);
}

for each pixel
{
    compute ray starting point and direction;
    illumination = TraceRay(ray, 0) ;
    pixel color = illumination tone mapped to display range
}
*/

// #define ASPECT_RATIO 1.777778 // 16/9
// #define PICTURES_AMMOUNT 10
// #define MIN_WIDTH 800
// #define MAX_WIDTH 1920
// #define REFLECT_RECURESION_LIMIT 5

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define CHANNELS 3

__global__ void pixel_color(float *pixels)
{
    if (blockIdx.y < IMAGE_HEIGHT && blockIdx.x < IMAGE_WIDTH)
    {
        int color_channel = threadIdx.x;
        int pixel_pos = (gridDim.x * blockIdx.y + blockIdx.x) + (color_channel * gridDim.x * gridDim.y);
        float color = 0.0f;
        if (color_channel == 0)
            color = (float)blockIdx.x / (float)IMAGE_WIDTH;
        else if (color_channel == 1)
            color = (float)blockIdx.y / (float)IMAGE_HEIGHT;
        pixels[pixel_pos] = color;
    }
}

int main()
{
    Image img(IMAGE_WIDTH, IMAGE_HEIGHT);

    float *d_pixels = NULL;
    float *h_pixels = (float *)malloc(CHANNELS * sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pixels, CHANNELS * sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH));

    dim3 gridDim(IMAGE_WIDTH, IMAGE_HEIGHT);
    dim3 blockDim(3);
    pixel_color<<<gridDim, blockDim>>>(d_pixels);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_pixels, d_pixels, CHANNELS * sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH, cudaMemcpyDeviceToHost));

    for (int y = 0; y < IMAGE_HEIGHT; y++)
    {
        for (int x = 0; x < IMAGE_WIDTH; x++)
        {
            RGBApixel color;
            color.Alpha = 0.0f;
            color.Red = h_pixels[(y * IMAGE_WIDTH + x) + (0 * IMAGE_HEIGHT * IMAGE_WIDTH)] * 255;
            color.Green = h_pixels[(y * IMAGE_WIDTH + x) + (1 * IMAGE_HEIGHT * IMAGE_WIDTH)] * 255;
            color.Blue = h_pixels[(y * IMAGE_WIDTH + x) + (2 * IMAGE_HEIGHT * IMAGE_WIDTH)] * 255;
            img.write_pixel(x, y, color);
        }
    }

    img.save();
}