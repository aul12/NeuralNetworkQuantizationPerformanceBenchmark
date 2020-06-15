/**
 * @file OpenCV.hpp
 * @author paul
 * @date 01.12.19
 * @brief OpenCV @TODO
 */

#ifndef TFBENCHMARK_OPENCV_HPP
#define TFBENCHMARK_OPENCV_HPP

#include <string>
#include <opencv2/opencv.hpp>

class OpenCV {
public:
    explicit OpenCV(const std::string &fname);

    auto forward(const cv::Mat &data) -> std::array<double, 30>;

private:
    cv::dnn::Net net;
};


#endif //TFBENCHMARK_OPENCV_HPP
