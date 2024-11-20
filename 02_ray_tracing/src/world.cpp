#ifndef WORLDH
#define WORLDH

#include "sphere.cpp"
#include "camera.cpp"

class world
{
public:
    __device__ world(
        sphere **d_spheres, int spheres_ammount,
        light **d_light, int light_ammount,
        camera *d_camera)
    {
        spheres = d_spheres;
        spheres_count = spheres_ammount;

        light = d_light;
        light_count = light_ammount;

        cam = d_camera;
    }

    camera *cam;

    sphere **spheres;
    int spheres_count;

    vec3 **light;
    int light_count;
};

#endif
