#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "cuda_utils.h"
#include "image.h"
#include "scene.h"
#include "light.h"

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
#define SAMPLES_AMMOUNT 100

#define MAX_RESOLUTION 1920
#define PICTURES_AMMOUNT 5

#define SKY vec3(0.6f, 0.7f, 0.8f)

#define RED 0
#define GREEN 1
#define BLUE 2

#define MAX_RECURSION_DEPTH 5

__device__ vec3 color(const ray &r, scene *scene, curandState *local_rand_state, int depth)
{
    vec3 illumination = vec3(1.0, 1.0, 1.0);
    if (depth == MAX_RECURSION_DEPTH)
        return illumination;

    hit_record rec;
    if (scene->spheres->hit(r, 0.001f, FLT_MAX, rec))
    {
        ray scattered;
        vec3 attenuation;

        hit_record light_rec;

        float light_intensity = SKY.squared_length();
        for (int i = 0; i < scene->lights->list_size; i++)
        {
            light *l = (light *)scene->lights->list[i];
            light_intensity += l->intensity(r);
        }

        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state))
        {
            if (light_intensity > 1.0f)
                light_intensity = 1.0f;
            illumination *= attenuation * light_intensity;
            illumination *= color(scattered, scene, local_rand_state, depth + 1);
        }
    }
    else
    {
        vec3 unit_direction = unit_vector(r.direction());
        float t = (0.5f * unit_direction.y()) + 0.5f;
        vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * SKY;
        illumination *= c;
    }

    return illumination;
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_pos = j * max_x + i;
    curand_init(52, pixel_pos, 0, &rand_state[pixel_pos]);
}

__global__ void render(vec3 *fb, int image_width, int image_height, int ns, scene **scene, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height))
        return;
    int pixel_pos = j * image_width + i;
    curandState local_rand_state = rand_state[pixel_pos];

    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(image_width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(image_height);
        ray r = (*scene)->cam->get_ray(u, v);
        col += color(r, *scene, &local_rand_state, 0);
    }
    rand_state[pixel_pos] = local_rand_state;
    col /= float(ns);
    col[RED] = sqrt(col[RED]);
    col[GREEN] = sqrt(col[GREEN]);
    col[BLUE] = sqrt(col[BLUE]);
    fb[pixel_pos] = col;
}

__global__ void create_world(
    hitable **d_spheres,
    hitable **d_lights,
    camera **d_camera,
    scene **d_scene)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_spheres[0] = new sphere(
            vec3(0, 0, 0), 0.5,
            new lambertian(vec3(0.1, 0.2, 0.5)));
        d_spheres[1] = new sphere(
            vec3(0, -100.5, -1), 100,
            new lambertian(vec3(0.2, 0.2, 0.2)));
        d_spheres[2] = new sphere(
            vec3(0, 0, -1.), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_spheres[3] = new sphere(
            vec3(-1.02, 0, -1.02), 0.1,
            new metal(vec3(0.2, 0.6, 0.6), 0.0));
        d_spheres[4] = new sphere(
            vec3(0, 0, -2), 0.45,
            new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_spheres[5] = new sphere(
            vec3(-1, 0, 0), 0.45,
            new metal(vec3(1.0f), 0.0));

        d_lights[0] = new light(vec3(1.0f, 4.0f, 0.0f));
        d_lights[1] = new light(vec3(-3.0f, -4.0f, 0.0f));

        *d_camera = new camera(
            vec3(-3.0f, 2.0f, 0),
            vec3(0, 0, 0),
            vec3(0, 1, 0),
            45.0f,
            16.0f / 9.0f);

        *d_scene = new scene();
        (*d_scene)->cam = *d_camera;
        (*d_scene)->spheres = new hitable_list(d_spheres, 6);
        (*d_scene)->lights = new hitable_list(d_lights, 2);
    }
}

__global__ void free_world(scene **sc)
{

    for (int i = 0; i < (*sc)->spheres->list_size; i++)
    {
        delete ((sphere *)(*sc)->spheres->list[i])->mat_ptr;
        delete (*sc)->spheres->list[i];
    }
    for (int i = 0; i < (*sc)->lights->list_size; i++)
    {
        delete (*sc)->lights->list[i];
    }
    delete (*sc)->cam;
    delete (*sc)->spheres;
    delete (*sc)->lights;
}

int main()
{

    clock_t start, stop;
    int image_width = 800;
    int step = (MAX_RESOLUTION - image_width) / PICTURES_AMMOUNT;
    int samples_number = SAMPLES_AMMOUNT;
    for (; image_width <= 1920; image_width += step)
    {
        int image_height = image_width / (16.0f / 9.0f);
        int tx = 8;
        int ty = 8;

        printf("Rendering %dx%d by %d samples\n", image_width, image_height, samples_number);

        int num_pixels = image_width * image_height;
        size_t fb_size = num_pixels * sizeof(vec3);

        // Pixels Allocation
        vec3 *fb;
        CHECK_CUDA_ERROR(cudaMallocManaged((void **)&fb, fb_size));

        // World set
        curandState *d_rand_state;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

        hitable **d_spheres;
        hitable **d_lights;
        camera **d_camera;
        scene **d_scene;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_spheres, 6 * sizeof(hitable *)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_lights, 2 * sizeof(hitable *)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_camera, sizeof(camera *)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_scene, sizeof(scene *)));
        create_world<<<1, 1>>>(d_spheres, d_lights, d_camera, d_scene);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // ---- ----

        // Render
        start = clock();
        dim3 blocks(image_width / tx + 1, image_height / ty + 1);
        dim3 threads(tx, ty);
        render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        render<<<blocks, threads>>>(fb, image_width, image_height, samples_number, d_scene, d_rand_state);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        printf("Took render %lf seconds\n", timer_seconds);
        // ---- ----

        Image img(image_width, image_height);
        img.set_exec_time(timer_seconds);
        for (int y = image_height - 1; y >= 0; y--)
        {
            for (int x = 0; x < image_width; x++)
            {
                size_t pixel_index = y * image_width + x;
                vec3 pixel = fb[pixel_index];
                img.write_vec3(x, y, pixel);
            }
        }
        img.save();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        free_world<<<1, 1>>>(d_scene);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaFree(d_spheres));
        CHECK_CUDA_ERROR(cudaFree(d_lights));
        CHECK_CUDA_ERROR(cudaFree(d_camera));
        CHECK_CUDA_ERROR(cudaFree(d_scene));
        CHECK_CUDA_ERROR(cudaFree(d_rand_state));
        CHECK_CUDA_ERROR(cudaFree(fb));
        CHECK_CUDA_ERROR(cudaDeviceReset());
    }
}
