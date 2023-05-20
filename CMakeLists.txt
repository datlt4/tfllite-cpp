cmake_minimum_required(VERSION 3.17)
project(CornerDetector)

option(DEBUG "DEBUG" OFF)

if (DEBUG)
    add_definitions(-DDEBUG)
endif (DEBUG)

set(CMAKE_CXX_STANDARD 14)

set(SRC main.cpp)
set(SRC ${SRC} CornerDetector.cpp)

# OpenCV Integration
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
set(INC ${INC} "/root/tensorflow/")
set(INC ${INC} "/root/.cache/bazel/_bazel_root/efb88f6336d9c4a18216fb94287b8d97/external/flatbuffers/include")
add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION "/root/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so")
set(LIB ${LIB} tensorflowlite)

include_directories(${INC})
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${LIB})
