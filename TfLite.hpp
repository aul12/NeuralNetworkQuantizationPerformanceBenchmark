/**
 * @file TfLite.hpp
 * @author paul
 * @date 01.12.19
 * @brief TfLite @TODO
 */

#ifndef TFBENCHMARK_TFLITE_HPP
#define TFBENCHMARK_TFLITE_HPP


#include <string>
#include <memory>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <opencv2/core/mat.hpp>

class TfLite {
public:
    explicit TfLite(const std::string &fname, int numOfThreads, int size, int channels);

    auto forward(const cv::Mat &img) -> std::array<double, 30>;
private:
    std::vector<int64_t> tensorDims;

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};


#endif //TFBENCHMARK_TFLITE_HPP
