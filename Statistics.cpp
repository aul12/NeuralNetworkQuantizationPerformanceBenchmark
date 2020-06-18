/**
 * @file Statistics.cpp
 * @author paul
 * @date 08.06.20
 * Description here TODO
 */
#include <iostream>
#include <iomanip>

#include "Statistics.hpp"

void Statistics::submit(const std::array<double, 30> &predictions, std::size_t groundTruth) {
    std::lock_guard<std::mutex> lockGuard{mutex};
    results.emplace_back(predictions, groundTruth);
}


void Statistics::printConfusionMatrix() const {
    std::array<std::array<double, 30>, 30> confMatrix{};

    for (const auto &[pdf, label] : results) {
        auto prediction = std::distance(pdf.cbegin(), std::max_element(pdf.cbegin(), pdf.cend()));
        confMatrix[prediction][label] += 1;
    }

    std::cout << "GT\\Pred |\t";
    for (auto c=0; c<30; ++c) {
        std::cout << std::setw(4) << c << "|";
    }
    std::cout << "\n|";
    for (auto c=0; c<30; ++c) {
        std::cout << "---|";
    }

    for (auto y = 0u; y < confMatrix.size(); ++y) {
        std::cout << "\n|" << std::setw(7) << y << "|\t";
        for (auto x = 0u; x < confMatrix[x].size(); ++x) {
            std::cout << std::setw(4) << confMatrix[x][y] << "|";
        }
    }
}

void Statistics::printAccuracy() const {
    std::size_t correct = 0;

    for (const auto &[pdf, label] : results) {
        auto prediction = std::distance(pdf.cbegin(), std::max_element(pdf.cbegin(), pdf.cend()));
        if (label == static_cast<std::size_t>(prediction)) {
            correct += 1;
        }
    }

    std::cout << "Accuracy: " << static_cast<float>(correct) / results.size() * 100 << "%" << std::endl;
}

