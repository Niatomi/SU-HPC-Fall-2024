#include "image.cpp"
#include "sphere.cpp"
#include "vec3.cpp"
#include "ray.cpp"
#include "camera.cpp"
#include "world.cpp"

#include "cuda_utils.h"

#include <stdio.h>
#include <float.h>
#include <math.h>

#include "thrust/device_vector.h"
#include <curand_kernel.h>

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

// #define PICTURES_AMMOUNT 10
// #define MIN_WIDTH 800
// #define MAX_WIDTH 1920
// #define REFLECT_RECURESION_LIMIT 5

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

#define CHANNELS 3
#define RED 0
#define GREEN 1
#define BLUE 2

#define SKY vec3(0.6f, 0.7f, 0.8f)
// #define SKY vec3(0.0f);

__device__ color3 trace_ray(ray r, world **w, int depth)
{
    if ((*w)->spheres_count == 0 || depth == 5)
        return SKY;

    int sphere_idx = -1;
    float min_t = FLT_MAX;

    for (int i = 0; i < (*w)->spheres_count; i++)
    {
        sphere *sph = (*w)->spheres[i];
        vec3 oc = r.origin() - sph->position;
        float a = dot(r.direction(), r.direction());
        float b = 2.0f * dot(oc, r.direction());
        float c = dot(oc, oc) - sph->radius * sph->radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0)
            continue;
        float t = (b - sqrt(discr)) / (2.0f * a);
        if (t < min_t && t > 0.0f)
        {
            min_t = t;
            sphere_idx = i;
        }
    }

    if (sphere_idx == -1)
        return SKY;

    sphere *closest_sph = (*w)->spheres[sphere_idx];
    vec3 origin = r.origin() - closest_sph->position;
    vec3 hit_point = origin + r.direction() * min_t;
    vec3 point_norm = hit_point.normilize();

    color3 illumintaion = closest_sph->albedo;
    for (int i = 0; i < (*w)->light_count; i++)
    {
        light *light = (*w)->light[i];
        vec3 norm_light = light->normilize();
        float light_intensity = max(dot(hit_point, -norm_light), 0.0f);
        illumintaion *= light_intensity;
    }

    vec3 refelcted = reflect(r.direction(), point_norm);
    ray reflected(hit_point, point_norm);
    return illumintaion + trace_ray(reflected, w, depth + 1) * 0.3f;
}

__global__ void pixel_color(float *pixels, world **w)
{
    if (blockIdx.y < IMAGE_HEIGHT && blockIdx.x < IMAGE_WIDTH)
    {
        int color_channel = threadIdx.x;
        float x = blockIdx.x;
        float y = blockIdx.y;

        ray r = (*w)->cam->get_ray(x, y);
        vec3 color = trace_ray(r, w, 0);

        int pixel_pos = (gridDim.x * blockIdx.y + blockIdx.x);
        int pR = pixel_pos + (RED * gridDim.x * gridDim.y);
        int pG = pixel_pos + (GREEN * gridDim.x * gridDim.y);
        int pB = pixel_pos + (BLUE * gridDim.x * gridDim.y);

        pixels[pR] = color.e[RED];
        pixels[pG] = color.e[GREEN];
        pixels[pB] = color.e[BLUE];
    }
}

__global__ void world_init(
    sphere **d_spheres, int spheres_ammount,
    light **d_light, int light_ammount,
    world **d_world,
    camera **d_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_spheres[0] = new sphere(0.5f, vec3(-1.0f, 0.0f, 0.0f), color3(0.5f, 0.0f, 0.0f));
        d_spheres[1] = new sphere(0.5f, vec3(0.5f, 0.0f, 0.0f), color3(0.0f, 0.5f, 0.0f));
        d_spheres[2] = new sphere(0.5f, vec3(1.5f, 0.0f, 0.6f), color3(0.5f, 0.0f, 0.5));
        // d_spheres[3] = new sphere(100.0f, vec3(0.0f, -50.0f, 0.0f), color3(1.0f));
        d_light[0] = new light(5.0f);
        *d_camera = new camera();
        *d_world = new world(
            d_spheres, spheres_ammount,
            d_light, light_ammount,
            *d_camera);
    }
}

int main()
{

    int spheres_ammount = 3;
    int light_ammount = 1;

    // Generate world
    sphere **d_spheres = NULL;
    light **d_light = NULL;
    camera **d_camera = NULL;
    world **d_world = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_world, sizeof(world)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_camera, sizeof(camera)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_light, light_ammount * sizeof(light)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_spheres, spheres_ammount * sizeof(sphere)));
    world_init<<<1, 1>>>(
        d_spheres, spheres_ammount,
        d_light, light_ammount,
        d_world, d_camera);
    //---- ----

    // Render
    clock_t start, end;
    start = clock();
    float *d_pixels = NULL;
    float *h_pixels = (float *)malloc(CHANNELS * sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pixels, CHANNELS * sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH));
    dim3 gridDim(IMAGE_WIDTH, IMAGE_HEIGHT);
    dim3 blockDim(1);
    pixel_color<<<gridDim, blockDim>>>(d_pixels, d_world);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_pixels, d_pixels, CHANNELS * sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH, cudaMemcpyDeviceToHost));
    end = clock();
    double exec_time = double(end - start) / double(CLOCKS_PER_SEC);
    // ---- ----

    Image img(IMAGE_WIDTH, IMAGE_HEIGHT);
    img.set_exec_time(exec_time);
    for (int y = 0; y < IMAGE_HEIGHT; y++)
    {
        for (int x = 0; x < IMAGE_WIDTH; x++)
        {
            vec3 pixel_color(
                h_pixels[(y * IMAGE_WIDTH + x) + (RED * IMAGE_HEIGHT * IMAGE_WIDTH)],
                h_pixels[(y * IMAGE_WIDTH + x) + (GREEN * IMAGE_HEIGHT * IMAGE_WIDTH)],
                h_pixels[(y * IMAGE_WIDTH + x) + (BLUE * IMAGE_HEIGHT * IMAGE_WIDTH)]);
            pixel_color = pixel_color.clamp(vec3(0.0f), vec3(1.0f));
            RGBApixel color;
            color.Alpha = 0.0f;
            color.Red = pixel_color.r() * 255.0f;
            color.Green = pixel_color.g() * 255.0f;
            color.Blue = pixel_color.b() * 255.0f;
            img.write_pixel(x, y, color);
        }
    }
    img.save();
}
