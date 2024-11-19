#pragma once

#include "EasyBMP/EasyBMP.h"
#include <memory>

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

private:
    std::shared_ptr<BMP> img;
    char *path;

    int img_width;
    int img_height;
};
