#include <future>

#include "TfLite.hpp"
#include "OpenCV.hpp"
#include "Statistics.hpp"
#include "Loader.hpp"

void work(Loader &loader, std::atomic_int &count, std::shared_ptr<TfLite> tfLite, std::shared_ptr<OpenCV> openCv,
          Statistics &tfStatistics, Statistics &cvStatistics) {
    while (loader.hasNext()) {
        auto[img, label] = loader.next();
        auto tfResult = tfLite->forward(img);
        auto cvResult = openCv->forward(img);
        tfStatistics.submit(tfResult, label);
        cvStatistics.submit(cvResult, label);

        count.fetch_add(1);
        std::cout << "\r" << count << std::flush;
    }
}

int main(int argc, char **argv) {
    if (argc <= 3) {
        std::cerr << "Usage: " << argv[0] << " setup.json cnn.pb cnn.tflite" << std::endl;
        std::exit(1);
    }
    Loader loader{argv[1]};

    Statistics tfStatistics{}, cvStatistics{};

    std::atomic_int count = 0;

    std::vector<std::future<void>> futures;

    for (auto c = 0u; c < std::thread::hardware_concurrency() * 2; ++c) {
        auto tfLite = std::make_shared<TfLite>(argv[3], 1, 80, 3);
        auto openCv = std::make_shared<OpenCV>(argv[2]);
        futures.emplace_back(
                std::async(std::launch::async, [&loader, &count, tfLite, openCv, &tfStatistics, &cvStatistics]() {
                    work(loader, count, tfLite, openCv, tfStatistics, cvStatistics);
                }));
    }

    for (const auto &future : futures) {
        if (future.valid()) {
            future.wait();
        }
    }
    std::cout << std::endl;

    std::cout << "With Quantization: " << std::endl;
    tfStatistics.printAll<10>();

    std::cout << "\n\n\nWithout Quantization: " << std::endl;
    cvStatistics.printAll<10>();

    return 0;
}
