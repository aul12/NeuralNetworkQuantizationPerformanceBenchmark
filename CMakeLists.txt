cmake_minimum_required(VERSION 3.15)
project(QuantizationPerformance)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror -march=native -mtune=native")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb -O0")
else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")
endif ()

find_package(OpenCV REQUIRED)

set(SOURCES main.cpp OpenCV.cpp TfLite.cpp Statistics.cpp Loader.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        libtensorflow-lite.a
        edgetpu
        pthread
        )
