#ifndef CAMERAH
#define CAMERAH

#include "ray.cpp"

#define ASPECT_RATIO 16.0f / 9.0f

class camera
{
public:
    __device__ camera()
    {
        cam_center = vec3(0.0, 0.0, 0.0f);
        float aspect_ratio = ASPECT_RATIO;
        image_width = 1920;
        focal_length = 1.0f;
        image_height = int(image_width / aspect_ratio);
        init_vars();
    }
    __device__ camera(vec3 origin)
    {
        cam_center = origin;
        float aspect_ratio = ASPECT_RATIO;
        image_width = 1920;
        focal_length = 0.5f;
        image_height = int(image_width / aspect_ratio);
        init_vars();
    }
    __device__ camera(vec3 origin, int img_width)
    {
        cam_center = origin;
        float aspect_ratio = ASPECT_RATIO;
        image_width = img_width;
        focal_length = 0.5f;
        image_height = int(image_width / aspect_ratio);
        init_vars();
    }

    __device__ void init_vars()
    {
        // Camera
        float viewport_height = 2.0f;
        float viewport_width = viewport_height * (float(image_width) / (float)image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = vec3(viewport_width, 0, 0);
        vec3 viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        // vec3 viewport_left_down = cam_center + vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        vec3 viewport_upper_left = cam_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    __device__ ray get_ray(int x, int y)
    {
        vec3 pixel_viewport_pos = pixel00_loc + (x * pixel_delta_u) + (y * pixel_delta_v);
        vec3 ray_direction = pixel_viewport_pos - cam_center;
        return ray(cam_center, ray_direction);
    }

    vec3 cam_center;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    float focal_length;

    float aspect_ratio;
    int image_height;
    int image_width;
};

#endif
