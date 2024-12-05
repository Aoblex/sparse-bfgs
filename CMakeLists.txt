cmake_minimum_required(VERSION 3.20)

# Project name
project(BFGS)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable
add_executable(
    BFGS
    src/main.cpp
    src/BFGS.cpp
    src/LineSearch.cpp
    src/MatrixOps.cpp
)

# Add include directory
target_include_directories(
    BFGS
    PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
