/**
 * @file TfLite.cpp
 * @author paul
 * @date 01.12.19
 * @brief TfLite @TODO
 */

#include <iostream>
#include "TfLite.hpp"

TfLite::TfLite(const std::string &fname, int numOfThreads, int size, int channels) : tensorDims{1,size, size,channels},
    model{tflite::FlatBufferModel::BuildFromFile(fname.c_str())} {

    tflite::ops::builtin::BuiltinOpResolver resolver;

    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
    }

    interpreter->SetNumThreads(numOfThreads);

    interpreter->AllocateTensors();

    if (model == nullptr) {
        throw std::runtime_error{"Failed to build FlatBufferModel from file"};
    }

    interpreter->Invoke();
}

auto TfLite::forward(const cv::Mat &img) -> std::array<double, 30> {
    const auto inputSize = tensorDims[0] * tensorDims[1] * tensorDims[2] * tensorDims[3];

    auto *const input = interpreter->typed_input_tensor<float>(0);
    for (int64_t c = 0; c < inputSize; c += 3) {
        const auto *const pix = (img.data + c);
        input[c + 0] = pix[2]; // RGB to BGR
        input[c + 1] = pix[1];
        input[c + 2] = pix[0];
    }
    interpreter->Invoke();

    const auto *const output = interpreter->typed_output_tensor<float>(0);

    std::array<double, 30> ret{};
    std::copy(output, output+30, ret.begin());
    return ret;
}
