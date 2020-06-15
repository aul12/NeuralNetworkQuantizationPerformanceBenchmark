/**
 * @file Loader.hpp
 * @author paul
 * @date 08.06.20
 * Description here TODO
 */
#ifndef QUANTIZATIONPERFORMANCE_LOADER_HPP
#define QUANTIZATIONPERFORMANCE_LOADER_HPP

#include <string>
#include <set>
#include <filesystem>
#include <mutex>

#include <opencv2/core/mat.hpp>

class Loader {
    public:
        explicit Loader(const std::string &configPath);

        auto next() -> std::pair<cv::Mat, std::size_t>;

        [[nodiscard]] auto hasNext() const -> bool;

        static auto numberToLabel(std::size_t label) -> std::string;
    private:
        void step();

        std::set<std::string> recordings;

        decltype(recordings)::const_iterator recordingsIt;
        std::map<std::string, std::string>::const_iterator directoryIt;
        std::filesystem::directory_iterator fileIter;

        std::size_t countCreated;

        mutable std::mutex lock;
};


#endif //QUANTIZATIONPERFORMANCE_LOADER_HPP
