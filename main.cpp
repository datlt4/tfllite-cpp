#include "CornerDetector.h"
#include <vector>

int main(int argc, char **argv)
{
    // cornerRefiner Model
    std::string cornerRefinerModelPath = "../models/cornerRefiner.tflite";
    // getCorners Model
    std::string getCornersModelPath = "../models/getCorners.tflite";

    std::unique_ptr<CornerDetector> cornerDetector = std::make_unique<CornerDetector>(CornerDetector(cornerRefinerModelPath, getCornersModelPath));
    // example image
    std::vector<std::string> imagePaths;

    int i = 1;
    do
    {
        cv::Mat image = cv::imread(argc == 1 ? "../asset/paper1.png" : argv[i]);
#ifdef DEBUG
        std::vector<float> vecTimePoint;
        for (int i = 0; i < 1000; ++i)
        {
            std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
#endif // DEBUG
            cornerDetector->inference(image);
#ifdef DEBUG
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            vecTimePoint.push_back(duration);
        }
        std::cout << "[ LOG ] Average time ellapsed:" << std::accumulate(vecTimePoint.begin(), vecTimePoint.end(), 0, [](int64_t x, int64_t y)
                                                                         { return x + y; }) /
                                                             vecTimePoint.size()
                  << std::endl;
#endif // DEBUG
        cv::imwrite("out" + std::to_string(i) + ".png", image);
        ++i;
    } while (i < argc);

    return 0;
}

// ./CornerDetector ../asset/paper1.png ../asset/paper2.png
