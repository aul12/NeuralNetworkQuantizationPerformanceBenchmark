/**
 * @file Loader.cpp
 * @author paul
 * @date 08.06.20
 * Description here TODO
 */
#include <opencv2/core/persistence.hpp>
#include <map>
#include <opencv2/opencv.hpp>
#include "Loader.hpp"

std::map<std::string, std::size_t> labelNumber = {
        {"CROSSWALK",         0},
        {"STOPLINE",          1},
        {"GIVE_WAY_LINE",     2},
        {"R_ARROW",           3},
        {"L_ARROW",           4},
        {"FORBIDDEN",         5},
        {"SPEEDLIMIT10",      6},
        {"SPEEDLIMIT20",      7},
        {"SPEEDLIMIT30",      8},
        {"SPEEDLIMIT40",      9},
        {"SPEEDLIMIT50",      10},
        {"SPEEDLIMIT60",      11},
        {"SPEEDLIMIT70",      12},
        {"SPEEDLIMIT80",      13},
        {"SPEEDLIMIT90",      14},
        {"NSPEEDLIMIT",       15},
        {"STARTLINE_PARKING", 16},
        {"CROSSINGLINE",      17},
        {"PEDESTRIAN",        18},
        {"EXPRESS_START",     19},
        {"EXPRESS_END",       20},
        {"NO_PASSING_START",  21},
        {"NO_PASSING_END",    22},
        {"UPHILL",            23},
        {"DOWNHILL",          24},
        {"PEDESTRIAN_ISLAND", 25},
        {"SHARP_TURN_RIGHT",  26},
        {"SHARP_TURN_LEFT",   27},
        {"RIGHT_OF_WAY",      28},
        {"NO_CLASS",          29}
};

std::map<std::string, std::string> directories = {
        {"BarredArea",       "FORBIDDEN"},
        {"Downhill",         "DOWNHILL"},
        {"ExpressEnd",       "EXPRESS_END"},
        {"ExpressStart",     "EXPRESS_START"},
        {"GiveWaySigns",     "GIVE_WAY_LINE"},
        {"LeftArrowSigns",   "L_ARROW"},
        {"NoPassingEnd",     "NO_PASSING_END"},
        {"NoPassingStart",   "NO_PASSING_START"},
        {"Pedestrian",       "PEDESTRIAN"},
        {"PedestrianIsland", "PEDESTRIAN_ISLAND"},
        {"RightArrowSigns",  "R_ARROW"},
        {"RightOfWay",       "RIGHT_OF_WAY"},
        {"SharpTurnLeft",    "SHARP_TURN_LEFT"},
        {"SharpTurnRight",   "SHARP_TURN_RIGHT"},
        {"Speed10EndSigns",  "NSPEEDLIMIT"},
        {"Speed10Signs",     "SPEEDLIMIT10"},
        {"Speed20EndSigns",  "NSPEEDLIMIT"},
        {"Speed20Signs",     "SPEEDLIMIT20"},
        {"Speed30EndSigns",  "NSPEEDLIMIT"},
        {"Speed30Signs",     "SPEEDLIMIT30"},
        {"Speed40EndSigns",  "NSPEEDLIMIT"},
        {"Speed40Signs",     "SPEEDLIMIT40"},
        {"Speed50EndSigns",  "NSPEEDLIMIT"},
        {"Speed50Signs",     "SPEEDLIMIT50"},
        {"Speed60EndSigns",  "NSPEEDLIMIT"},
        {"Speed60Signs",     "SPEEDLIMIT60"},
        {"Speed70EndSigns",  "NSPEEDLIMIT"},
        {"Speed70Signs",     "SPEEDLIMIT70"},
        {"Speed80EndSigns",  "NSPEEDLIMIT"},
        {"Speed80Signs",     "SPEEDLIMIT80"},
        {"Speed90EndSigns",  "NSPEEDLIMIT"},
        {"Speed90Signs",     "SPEEDLIMIT90"},
        {"StartlineParking", "STARTLINE_PARKING"},
        {"StopLineSigns",    "STOPLINE"},
        {"Uphill",           "UPHILL"},
        {"XingSigns",        "CROSSWALK"},
        {"NoClass",          "NO_CLASS"}
};


Loader::Loader(const std::string &configPath) : countCreated{0} {
    cv::FileStorage fileStorage{configPath, 0};
    auto rootPath = fileStorage["TrainingData"].string();
    auto recordingNode = fileStorage["Recordings"];
    for (auto &&c : recordingNode) {
        recordings.emplace(rootPath + c.string());
    }

    recordingsIt = recordings.cbegin();
    directoryIt = directories.cbegin();
    fileIter = std::filesystem::directory_iterator{*recordingsIt + "/" + directoryIt->first};
}

auto Loader::next() -> std::pair<cv::Mat, std::size_t> {
    cv::Mat img;
    std::size_t label;

    do {
        lock.lock();
        auto fname = fileIter->path();
        label = labelNumber[directoryIt->second];

        step();

        lock.unlock();

        img = cv::imread(fname, cv::IMREAD_COLOR);
    } while (img.rows == 0 || img.cols == 0);

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    countCreated += 1;
    return {img, label};
}

auto Loader::hasNext() const -> bool {
    if (countCreated > 4000) {
        return false;
    }
    std::lock_guard<std::mutex> guard{lock};
    return recordingsIt != recordings.cend();
}

auto Loader::numberToLabel(std::size_t label) -> std::string {
    for (const auto &[l, n] : labelNumber) {
        if (n == label) {
            return l;
        }
    }
    return "ERROR";
}

void Loader::step() {
    fileIter++;
    if (fileIter == std::filesystem::directory_iterator()) {
        do {
            directoryIt++;
        } while (!std::filesystem::exists(*recordingsIt + "/" + directoryIt->first));

        fileIter = std::filesystem::directory_iterator{*recordingsIt + "/" + directoryIt->first};

        if (directoryIt == directories.cend()) {
            if (recordingsIt != recordings.cend()) {
                recordingsIt++;
            }

            directoryIt = directories.cbegin();
        }
    }
}
