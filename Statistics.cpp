/**
 * @file Statistics.cpp
 * @author paul
 * @date 08.06.20
 * Description here TODO
 */
#include <iostream>
#include <iomanip>
#include <cmath>

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

    std::cout << "\n\n|GT\\Pred |\t";
    for (auto c=0; c<30; ++c) {
        std::cout << std::setw(4) << c << "|";
    }
    std::cout << "\n|---|";
    for (auto c=0; c<30; ++c) {
        std::cout << "---|";
    }

    for (auto y = 0u; y < confMatrix.size(); ++y) {
        std::cout << "\n|" << std::setw(7) << y << "|\t";
        for (auto x = 0u; x < confMatrix[x].size(); ++x) {
            std::cout << std::setw(4) << confMatrix[x][y] << "|";
        }
    }
    std::cout << "\n" << std::endl;
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

void Statistics::printNormalizedEntropy() const {
    double entropySum = 0;
    for (const auto &[pdf, _] : results) {
        double entropy = 0;
        for (const auto &elem : pdf) {
            if (elem > 0) {
                entropy += elem * std::log2(elem);
            }
        }
        entropy = (-entropy) / std::log2(pdf.size());
        entropySum += entropy;
    }

    std::cout << "Normalized Entropy: " << entropySum / results.size() << std::endl;
}

