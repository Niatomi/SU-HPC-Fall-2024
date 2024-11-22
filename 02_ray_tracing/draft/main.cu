#include <stdio.h>
// #include "ray.h"
#include "image.h"
#include <vector>
#include <math.h>
#include <glm/glm.hpp>

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
#define MAX_DEPTH 5

#include <random>
#include <ctime>

float random_float(float min, float max)
{

    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

typedef struct sphere
{
    float radius;
    glm::vec3 position;
    glm::vec3 albedo;
};

typedef struct camera
{
    float ax;
    float ay;
    float az;

    float bx;
    float by;
    float bz;
};

typedef struct ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

glm::vec4 trace_ray(ray ray, std::vector<sphere> spheres, int depth, int prev_sphere)
{
    glm::vec4 pixel_color{0.6f, 0.7f, 0.9f, 0.0f};
    if (spheres.size() == 0 || depth == MAX_DEPTH)
        return pixel_color;

    int closest_sphere_idx = -1;
    float min_t = FLT_MAX;

    for (int i = 0; i < spheres.size(); i++)
    {
        sphere sph = spheres[i];
        glm::vec3 origin = ray.origin - sph.position;
        double a = glm::dot(ray.direction, ray.direction);
        double b = 2.0f * glm::dot(origin, ray.direction);
        double c = glm::dot(origin, origin) - sph.radius * sph.radius;

        double discr = b * b - 4.0f * a * c;
        if (discr < 0)
            continue;

        // float t0 = (-b + glm::sqrt(discr)) / (2.0f * a);
        float t1 = (-b - glm::sqrt(discr)) / (2.0f * a);

        if (t1 > 0.0f && t1 < min_t)
        {
            min_t = t1;
            closest_sphere_idx = i;
        }
    }

    if (closest_sphere_idx == -1)
        return pixel_color;
    sphere closest_sphere = spheres[closest_sphere_idx];

    glm::vec3 origin = ray.origin - closest_sphere.position;
    glm::vec3 hit_point = origin + ray.direction * min_t;
    glm::vec3 norm = glm::normalize(hit_point);

    glm::vec3 light_source = glm::normalize(glm::vec3(-1.0f, -0.5f, 1));
    float light_intensity = glm::max(glm::dot(norm, -light_source), 0.0f); // == cos(angle)

    pixel_color.x = closest_sphere.albedo.x;
    pixel_color.y = closest_sphere.albedo.y;
    pixel_color.z = closest_sphere.albedo.z;

    // ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;
    // ray.Direction = glm::reflect(ray.Direction, payload.WorldNormal);
    ray.origin = closest_sphere.position + norm;
    ray.direction = glm::reflect(ray.direction, norm) + 0.1f * 0.5f;
    ray.direction = spheres[0].position;

    return pixel_color + trace_ray(ray, spheres, depth + 1, closest_sphere_idx) * 0.3f;
}

int main()
{
    Image img(IMAGE_WIDTH, IMAGE_HEIGHT);

    srand(time(NULL));
    camera cam;
    cam.ax = 8;
    cam.ay = 8;
    cam.az = -4;

    // cam.bx = 0.0f;
    // cam.by = 0.0f;
    cam.bz = 0.5f;

    float step = 0.5f;
    std::vector<sphere> spheres;
    sphere sph;
    sph.radius = 1.0f;
    sph.position = glm::vec3{0.0f, 0.0f, 2.0f};
    sph.albedo = glm::vec3(1.0f, 1.0f, 1.0f);
    spheres.push_back(sph);

    sph.radius = 1.0f;
    sph.position = glm::vec3{0, 0, 5};
    sph.albedo = glm::vec3(1.0f, 0.0f, 0.0f);
    spheres.push_back(sph);

    sph.radius = 1.0f;
    sph.position = glm::vec3{2, 0, 7};
    sph.albedo = glm::vec3(0.0f, 1.0f, 0.0f);
    spheres.push_back(sph);

    sph.radius = 1.0f;
    sph.position = glm::vec3{0, 3, 2.0f};
    sph.albedo = glm::vec3(0.0f, 0.0f, 1.0f);
    spheres.push_back(sph);

    // floot
    sph.radius = 1.0f;
    sph.position = glm::vec3{0, 0, 5};
    sph.albedo = glm::vec3(1.0f, 0.0f, 0.0f);
    spheres.push_back(sph);

    for (int y = 0; y < IMAGE_HEIGHT; y++)
    {
        for (int x = 0; x < IMAGE_WIDTH; x++)
        {
            glm::vec2 coord = {
                (float)x / IMAGE_WIDTH,
                (float)y / IMAGE_HEIGHT};
            coord = coord * 2.0f - 1.0f;

            ray ray;
            ray.origin = glm::vec3{cam.ax, cam.ay, cam.az};
            ray.direction = glm::vec3{coord.x, coord.y, cam.bz};

            glm::vec4 color_vec = trace_ray(ray, spheres, 0, -1);
            img.write_vec(x, y, color_vec);
        }
    }

    img.save();
}