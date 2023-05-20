#include "CornerDetector.h"

CornerDetector::CornerDetector(std::string cornerRefinerModelPath, std::string getCornersModelPath)
{
    // cornerRefinerModel
    this->cornerRefinerModel = tflite::FlatBufferModel::BuildFromFile(cornerRefinerModelPath.c_str());
    if (this->cornerRefinerModel == nullptr)
    {
        std::cerr << "[ ERROR ][ cornerRefiner ] Failed to load model" << std::endl;
        exit(-1);
    }
    tflite::InterpreterBuilder(*this->cornerRefinerModel.get(), this->cornerRefinerResolver)(&this->cornerRefinerInterpreter);
    if (this->cornerRefinerInterpreter == nullptr)
    {
        std::cerr << "[ ERROR ][ cornerRefiner ] Failed to initiate the interpreter" << std::endl;
        exit(-1);
    }
    if (this->cornerRefinerInterpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "[ ERROR ][ cornerRefiner ] Failed to allocate tensor" << std::endl;
        exit(-1);
    }
    this->cornerRefinerInterpreter->SetAllowFp16PrecisionForFp32(true);
    this->cornerRefinerInterpreter->SetNumThreads(1);
    // Get Input Tensor Dimensions
    this->cornerRefinerInput = this->cornerRefinerInterpreter->inputs()[0];
    this->cornerRefinerHeight = this->cornerRefinerInterpreter->tensor(this->cornerRefinerInput)->dims->data[1];
    this->cornerRefinerWidth = this->cornerRefinerInterpreter->tensor(this->cornerRefinerInput)->dims->data[2];
    this->cornerRefinerChannels = this->cornerRefinerInterpreter->tensor(this->cornerRefinerInput)->dims->data[3];
    // Get Output Tensor Dimensions
    this->cornerRefinerOutput = this->cornerRefinerInterpreter->outputs()[0];
    this->cornerRefinerOutputDims = std::make_unique<TfLiteIntArray>(*this->cornerRefinerInterpreter->tensor(this->cornerRefinerOutput)->dims);
    this->cornerRefinerOutputSize = this->cornerRefinerOutputDims->data[this->cornerRefinerOutputDims->size - 1];

    // getCornersModel
    this->getCornersModel = tflite::FlatBufferModel::BuildFromFile(getCornersModelPath.c_str());
    if (this->getCornersModel == nullptr)
    {
        std::cerr << "[ ERROR ][ getCorners ] Failed to load model" << std::endl;
        exit(-1);
    }
    tflite::InterpreterBuilder(*this->getCornersModel.get(), this->getCornersResolver)(&this->getCornersInterpreter);
    if (this->getCornersInterpreter == nullptr)
    {
        std::cerr << "[ ERROR ][ getCorners ] Failed to initiate the interpreter" << std::endl;
        exit(-1);
    }
    if (this->getCornersInterpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "[ ERROR ][ getCorners ] Failed to allocate tensor" << std::endl;
        exit(-1);
    }
    // Get Input Tensor Dimensions
    this->getCornersInput = this->getCornersInterpreter->inputs()[0];
    this->getCornersHeight = this->getCornersInterpreter->tensor(this->getCornersInput)->dims->data[1];
    this->getCornersWidth = this->getCornersInterpreter->tensor(this->getCornersInput)->dims->data[2];
    this->getCornersChannels = this->getCornersInterpreter->tensor(this->getCornersInput)->dims->data[3];
    // Get Output Tensor Dimensions
    this->getCornersOutput = this->getCornersInterpreter->outputs()[0];
    this->getCornersOutputDims = std::make_unique<TfLiteIntArray>(*this->getCornersInterpreter->tensor(this->getCornersOutput)->dims);
    this->getCornersOutputSize = this->getCornersOutputDims->data[this->getCornersOutputDims->size - 1];
}

void CornerDetector::getCornersInference(cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(this->getCornersWidth, this->getCornersHeight));
    resizedImage.convertTo(resizedImage, CV_32FC3);
    memcpy(reinterpret_cast<void *>(this->getCornersInterpreter->typed_input_tensor<float>(0)), reinterpret_cast<void *>(resizedImage.data), resizedImage.total() * resizedImage.elemSize());
    this->getCornersInterpreter->Invoke();
    float response[8];
    memcpy(reinterpret_cast<void *>(response), reinterpret_cast<void *>(this->getCornersInterpreter->typed_output_tensor<float>(0)), 8 * sizeof(float));
    for (int i = 0; i < 8; i = i + 2)
    {
        // response[i] *= image.cols;
        // response[i + 1] * image.rows;
        corners.push_back(cv::Point2f(response[i] * image.cols, response[i + 1] * image.rows));
    }
}

void CornerDetector::cornerRefinerInference(cv::Mat &image, cv::Point2f &corner)
{
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(this->cornerRefinerWidth, this->cornerRefinerHeight));
    resizedImage.convertTo(resizedImage, CV_32FC3);
    memcpy(reinterpret_cast<void *>(this->cornerRefinerInterpreter->typed_input_tensor<float>(0)), reinterpret_cast<void *>(resizedImage.data), resizedImage.total() * resizedImage.elemSize());
    this->cornerRefinerInterpreter->Invoke();
    float response[2];
    memcpy(reinterpret_cast<void *>(response), reinterpret_cast<void *>(this->cornerRefinerInterpreter->typed_output_tensor<float>(0)), 2 * sizeof(float));
    corner = cv::Point2f(response[0], response[1]);
}

void CornerDetector::inference(cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "[ ERROR ] Failed to load iamge" << std::endl;
        exit(-1);
    }
    std::vector<cv::Point2f> corners;
    this->getCornersInference(image, corners);
    std::vector<cv::Mat> imageAroundCorners;

    // roi
    cv::Rect2f roi;
    // Top-left
    roi = cv::Rect2f(
        cv::Point2f(
            std::max(0.0f, 2 * corners[0].x - (corners[1].x + corners[0].x) / 2),
            std::max(0.0f, 2 * corners[0].y - (corners[3].y + corners[0].y) / 2)),
        cv::Point2f(
            (corners[1].x + corners[0].x) / 2,
            (corners[3].y + corners[0].y) / 2));
    imageAroundCorners.push_back(cv::Mat(image, roi));
    // Top-right
    roi = cv::Rect2f(
        cv::Point2f(
            (corners[1].x + corners[0].x) / 2,
            std::max(0.0f, 2 * corners[1].y - (corners[1].y + corners[2].y) / 2)),
        cv::Point2f(
            std::min(static_cast<float>(image.cols - 1), corners[1].x + (corners[1].x - corners[0].x) / 2),
            (corners[1].y + corners[2].y) / 2));
    imageAroundCorners.push_back(cv::Mat(image, roi));
    // Bottom-right
    roi = cv::Rect2f(
        cv::Point2f(
            (corners[2].x + corners[3].x) / 2,
            (corners[1].y + corners[2].y) / 2),
        cv::Point2f(
            std::min(static_cast<float>(image.cols) - 1, corners[2].x + (corners[2].x - corners[3].x) / 2),
            std::min(static_cast<float>(image.rows) - 1, corners[2].y + (corners[2].y - corners[1].y) / 2)));
    imageAroundCorners.push_back(cv::Mat(image, roi));
    // Bottom-left
    roi = cv::Rect2f(
        cv::Point2f(
            std::max(0.0f, 2 * corners[3].x - (corners[2].x + corners[3].x) / 2),
            (corners[0].y + corners[3].y) / 2),
        cv::Point2f(
            (corners[3].x + corners[2].x) / 2,
            std::min(static_cast<float>(image.rows) - 1, corners[3].y + (corners[3].y - corners[0].y) / 2)));
    imageAroundCorners.push_back(cv::Mat(image, roi));

    // x_temp
    std::vector<float> x_temp{std::max(0.0f, 2 * corners[0].x - (corners[1].x + corners[0].x) / 2),
                                (corners[1].x + corners[0].x) / 2,
                                (corners[2].x + corners[3].x) / 2,
                                std::max(0.0f, 2 * corners[3].x - (corners[2].x + corners[3].x) / 2)};
    // y_temp
    std::vector<float> y_temp{std::max(0.0f, 2 * corners[0].y - (corners[3].y + corners[0].y) / 2),
                                std::max(0.0f, 2 * corners[1].y - (corners[1].y + corners[2].y) / 2),
                                (corners[1].y + corners[2].y) / 2,
                                (corners[0].y + corners[3].y) / 2};
    // Corner coordination
    std::vector<cv::Point2f> corner_address;
    for (int i = 0; i < 4; ++i)
    {
        cv::Mat img, _myImage;
        imageAroundCorners[i].copyTo(img);
        img.copyTo(_myImage);
        float ans_x = 0.0f, ans_y = 0.0f, x_start = 0.0f, y_start = 0.0f;
        cv::Size2f up_scale_factor(img.cols, img.rows);
        cv::Point2f y;
        while (_myImage.cols > 10 && _myImage.rows > 10)
        {
            cv::Point2f reponse, reponse_up;
            this->cornerRefinerInference(_myImage, reponse);
            reponse_up = cv::Point2f(reponse.x * up_scale_factor.width,
                                        reponse.y * up_scale_factor.height);
            y = reponse_up + cv::Point2f(x_start, y_start);
            float x_loc = y.x;
            float y_loc = y.y;
            float start_x, start_y;
            if (x_loc > _myImage.cols / 2)
                start_x = std::min(x_loc + _myImage.cols * CROP_FRAC / 2, static_cast<float>(_myImage.cols) * (1.0f - CROP_FRAC));
            else
                start_x = std::max(x_loc - _myImage.cols * CROP_FRAC / 2, 0.0f);
            if (y_loc > _myImage.rows / 2)
                start_y = std::min(y_loc + _myImage.rows * CROP_FRAC / 2, static_cast<float>(_myImage.rows) * (1.0f - CROP_FRAC));
            else
                start_y = std::max(y_loc - _myImage.rows * CROP_FRAC / 2, 0.0f);
            ans_x += start_x;
            ans_y += start_y;
            _myImage = cv::Mat(_myImage, cv::Rect2f(cv::Point2f(start_x, start_y), cv::Point2f(start_x + _myImage.cols * CROP_FRAC, start_y + _myImage.rows * CROP_FRAC)));
            img = cv::Mat(img, cv::Rect2f(cv::Point2f(start_x, start_y), cv::Point2f(start_x + img.cols * CROP_FRAC, start_y + img.rows * CROP_FRAC)));
            up_scale_factor = cv::Size2f(img.cols, img.rows);
        }
        corner_address.push_back(cv::Point2f(ans_x + y.x + x_temp[i], ans_y + y.y + y_temp[i]));
    }
    for (int i = 0; i < 4; ++i)
    {
        cv::line(image, corner_address[i % 4], corner_address[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        cv::circle(image, corner_address[i % 4], 5, cv::Scalar(0, 0, 255), cv::FILLED);

    }
}
