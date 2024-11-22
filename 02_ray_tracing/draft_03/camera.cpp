#ifndef CAMERAH
#define CAMERAH

#include "ray.cpp"

#define ASPECT_RATIO 16.0f / 9.0f

class camera
{
public:
    __device__ camera()
    {
        // vfov is top to bottom in degrees
        float vfov = 60.0f;
        float aspect = 16.0f / 9.0f;
        vec3 lookfrom(0.0f, 0.0f, 0.0f);
        vec3 lookat(0.0f, 0.0f, 1.0f);
        vec3 vup(0.0f, 1.0f, 0.0f);

        img_width = 1920;
        img_height = 1080;

        vec3 u, v, w;
        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        lower_left_corner = origin - half_width * u - half_height * v - w;
        horizontal = 2 * half_width * u;
        vertical = 2 * half_height * v;
    }

    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect)
    {
        // vfov is top to bottom in degrees
        vec3 u, v, w;
        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * u - half_height * v - w;
        horizontal = 2 * half_width * u;
        vertical = 2 * half_height * v;
    }

    __device__ ray get_ray(int x, int y)
    {
        return ray(origin, lower_left_corner + ((float)x / (float)img_width) * horizontal + ((float)y / (float)img_height) * vertical - origin);
    }
    __device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    int img_width;
    int img_height;
};

#endif
