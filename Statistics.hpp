/**
 * @file Statistics.hpp
 * @author paul
 * @date 08.06.20
 * Description here TODO
 */
#ifndef QUANTIZATIONPERFORMANCE_STATISTICS_HPP
#define QUANTIZATIONPERFORMANCE_STATISTICS_HPP

#include <array>
#include <deque>
#include <mutex>

class Statistics {
    public:
        void submit(const std::array<double, 30> &predictions, std::size_t groundTruth);

        template<std::size_t BINS>
        void printAll() const;

        void printConfusionMatrix() const;

        void printAccuracy() const;

        template<std::size_t BINS, std::size_t STEPS>
        void printOutputHist() const;

        template<std::size_t BINS, std::size_t STEPS>
        void printCertaintyHist() const;

        void printNormalizedEntropy() const;
    private:
        template<std::size_t BINS, std::size_t STEPS, typename T>
        static void printHist(std::array<T, BINS> hist);

        std::deque<std::pair<std::array<double, 30>, std::size_t>> results;
        std::mutex mutex;
};

template<std::size_t BINS>
void Statistics::printAll() const {
    printAccuracy();
    std::cout << std::endl << "Confusion Matrix:" << std::endl;
    printConfusionMatrix();
    printNormalizedEntropy();
    std::cout << std::endl << "Output value histogram:" << std::endl;
    printOutputHist<BINS, BINS>();
    std::cout << std::endl << "Certainty histogram:" << std::endl;
    printCertaintyHist<BINS, BINS>();
}

template<std::size_t BINS, std::size_t STEPS>
void Statistics::printOutputHist() const {
    std::array<float, BINS> hist{};
    for (const auto &[pdf, _] : results) {
        for (const auto &out : pdf) {
            auto binF = out * BINS;
            if (binF >= BINS) {
                binF = BINS - 1;
            } else if (binF < 0) {
                binF = 0;
            }
            hist[static_cast<std::size_t>(binF)] += 1;
        }
    }
    printHist<BINS, STEPS, typename decltype(hist)::value_type>(hist);
}

template<std::size_t BINS, std::size_t STEPS>
void Statistics::printCertaintyHist() const {
    std::array<float, BINS> hist{};
    for (const auto &[pdf, label] : results) {
        auto maxElem = std::max_element(pdf.cbegin(), pdf.cend());
        auto prediction = std::distance(pdf.cbegin(), maxElem);
        if (static_cast<std::size_t>(prediction) == label) {
            auto binF = *maxElem * BINS;
            if (binF >= BINS) {
                binF = BINS - 1;
            } else if (binF < 0) {
                binF = 0;
            }
            hist[static_cast<std::size_t>(binF)] += 1;
        }
    }
    printHist<BINS, STEPS, typename decltype(hist)::value_type>(hist);
}

template<std::size_t BINS, std::size_t STEPS, typename T>
void Statistics::printHist(std::array<T, BINS> hist) {
    T sum = 0;
    for (const auto &elem : hist) {
        sum += elem;
    }

    for (auto &elem : hist) {
        elem /= sum;
    }

    for (auto y = 0U; y < BINS; ++y) {
        auto val = 1 - (y + .5F) / BINS;
        for (auto x = 0U; x < hist.size(); ++x) {
            if (hist[x] > val) {
                std::cout << "   #\t";
            } else {
                std::cout << "    \t";
            }
        }
        std::cout << "\n";
    }

    for (auto x = 0U; x < BINS; ++x) {
        std::cout << std::setw(4) << static_cast<float>(x) / BINS << "\t";
    }

    std::cout << "\n\nAs Table:\n";
    for (auto c=0U; c<hist.size(); ++c) {
        auto low = static_cast<float>(c) / BINS;
        auto high = static_cast<float>(c+1) / BINS;
        std::cout << "\"[" << low << ", " << high << ")\":\t" << hist[c] << "," << std::endl;
    }
}

#endif //QUANTIZATIONPERFORMANCE_STATISTICS_HPP
