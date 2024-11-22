#ifndef SPHEREH
#define SPHEREH

#include "vec3.cpp"

class sphere
{
public:
    __device__ sphere()
    {
        radius = 0.5f;
        position = vec3(0.0f);
        albedo = vec3(1.0f);
    }
    __device__ sphere(float r)
    {
        radius = r;
        position = vec3(0.0f);
        albedo = vec3(1.0f);
    }
    __device__ sphere(vec3 pos)
    {
        radius = 0.5f;
        position = pos;
        albedo = vec3(1.0f);
    }
    __device__ sphere(float r, vec3 pos)
    {
        radius = r;
        position = pos;
        albedo = vec3(1.0f);
    }
    __device__ sphere(float r, vec3 pos, vec3 col)
    {
        radius = r;
        position = pos;
        albedo = col;
    }

    float radius;
    vec3 position;
    vec3 albedo;
};

#endif
