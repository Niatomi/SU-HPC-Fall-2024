#pragma once

#include "EasyBMP/EasyBMP.h"
#include <memory>

#include <glm/glm.hpp>

class Image
{
public:
    Image(int width, int height)
    {
        img = std::make_shared<BMP>();
        img->SetSize(width, height);
        img_height = height;
        img_width = width;
        path = "./image.bmp";
    }

    bool save()
    {
        return img->WriteToFile(path);
    }

    bool write_pixel(int x, int y, RGBApixel &pixel)
    {
        return img->SetPixel(x, img_height - y - 1, pixel);
    }

    bool write_vec(int x, int y, glm::vec4 vec)
    {
        vec = glm::clamp(vec, glm::vec4(0.0f), glm::vec4(1.0f));
        RGBApixel pixel;
        pixel.Alpha = vec.w;
        pixel.Red = vec.x * 255;
        pixel.Green = vec.y * 255;
        pixel.Blue = vec.z * 255;

        return write_pixel(x, y, pixel);
    }

private:
    std::shared_ptr<BMP> img;
    char *path;

    int img_width;
    int img_height;
};
