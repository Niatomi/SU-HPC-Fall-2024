#ifndef LIGHTH
#define LIGHTH

#include "hitable.h"

class light : public hitable
{
public:
    __device__ light() {}
    __device__ light(vec3 pos) : center(pos)
    {
        uv = unit_vector(center);
    }
    __device__ virtual float intensity(const ray &r) const;
    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;

    vec3 center;
    vec3 uv;
};

__device__ float light::intensity(const ray &r) const
{
    return max((dot(unit_vector(r.direction()), uv)), 0.0f);
}

__device__ bool light::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    return false;
}

#endif