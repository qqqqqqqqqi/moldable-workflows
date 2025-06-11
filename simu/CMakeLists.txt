cmake_minimum_required(VERSION 3.29)
project(simu)

set(CMAKE_CXX_STANDARD 20)

add_executable(simu main.cpp)

set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/libomp/19.1.6/lib/libomp.dylib")

include_directories("/opt/homebrew/opt/libomp/include")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(simu PUBLIC OpenMP::OpenMP_CXX)
else()
    message("OpenMP not found")
endif()
