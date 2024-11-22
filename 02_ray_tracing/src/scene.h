#ifndef SCENEH
#define SCENEH

#include "vec3.h"
#include "hitable_list.h"

class scene
{
public:
    __host__ __device__ scene() {}

    camera *cam;
    hitable_list *spheres;
    hitable_list *lights;
};

#endif
