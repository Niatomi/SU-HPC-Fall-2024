#ifndef IMAGEH
#define IMAGEH

#include "EasyBMP/EasyBMP.h"
#include <memory>
#include <filesystem>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

class Image
{
public:
    Image(int width, int height)
    {
        img = std::make_shared<BMP>();
        img->SetSize(width, height);
        img_height = height;
        img_width = width;
        path = "assets";
        full_path = path;
        full_path += "/";
        exec_time = 0.0f;

        full_path += std::to_string(img_width) + "x" + std::to_string(img_height) + "_";
    }

    void set_exec_time(float exec)
    {
        exec_time = exec;
    }

    bool save()
    {
        if (!fs::exists(path))
            fs::create_directories(path);

        full_path += std::to_string(0.0f);
        std::replace(full_path.begin(), full_path.end(), '.', '_');
        full_path += ".bmp";
        return img->WriteToFile(full_path.c_str());
    }

    bool write_pixel(int x, int y, RGBApixel &pixel)
    {
        return img->SetPixel(img_width - x - 1, y, pixel);
    }

private:
    std::shared_ptr<BMP> img;
    std::string path;
    std::string full_path;

    double exec_time;

    int img_width;
    int img_height;
};

#endif