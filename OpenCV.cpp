/**
 * @file OpenCV.cpp
 * @author paul
 * @date 01.12.19
 * @brief OpenCV @TODO
 */

#include "OpenCV.hpp"

OpenCV::OpenCV(const std::string &fname) : net{cv::dnn::readNetFromTensorflow(fname)} {}

auto OpenCV::forward(const cv::Mat &data) -> std::array<double, 30> {
    auto input = cv::dnn::blobFromImage(data, 1.0, data.size(), 0.0, true, CV_32F);
    net.setInput(input);

    auto result = net.forward();

    // The OpenCV Documentation does not define if the result of forward is of type float or double
    bool matIsDouble = (result.type() & CV_MAT_DEPTH_MASK) == CV_64F;

    std::array<double, 30> pdf{};
    for (auto c=0u; c<29; ++c) {
        if (matIsDouble) {
            pdf[c] = result.at<double>(c);
        } else {
            pdf[c] = result.at<float>(c);
        }
    }
    return pdf;
}
