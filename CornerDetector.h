
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"
#ifdef DEBUG
#include <numeric>
#include <chrono>
#endif // DEBUG

#ifdef __cplusplus
extern "C"{
#endif // __cplusplus

#define CROP_FRAC 0.85f

class CornerDetector
{
public:
    CornerDetector(std::string cornerRefinerModelPath, std::string getCornersModelPath);
    void getCornersInference(cv::Mat &image, std::vector<cv::Point2f> &corners);
    void cornerRefinerInference(cv::Mat &image, cv::Point2f &corner);
    void inference(cv::Mat &image);
private:
    std::unique_ptr<tflite::FlatBufferModel> cornerRefinerModel;
    std::unique_ptr<tflite::Interpreter> cornerRefinerInterpreter;
    tflite::ops::builtin::BuiltinOpResolver cornerRefinerResolver;
    int cornerRefinerInput;
    int cornerRefinerHeight;
    int cornerRefinerWidth;
    int cornerRefinerChannels;
    std::unique_ptr<TfLiteIntArray> cornerRefinerOutputDims;
    int cornerRefinerOutput;
    int cornerRefinerOutputSize;

    std::unique_ptr<tflite::FlatBufferModel> getCornersModel;
    std::unique_ptr<tflite::Interpreter> getCornersInterpreter;
    tflite::ops::builtin::BuiltinOpResolver getCornersResolver;
    int getCornersInput;
    int getCornersHeight;
    int getCornersWidth;
    int getCornersChannels;
    std::unique_ptr<TfLiteIntArray> getCornersOutputDims;
    int getCornersOutput;
    int getCornersOutputSize;
};

#ifdef __cplusplus
}
#endif // __cplusplus