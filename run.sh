#!/bin/bash

# If build/ exists, remove it
if [ -d "build" ]; then
  rm -rf build
fi

# Create build/ directory
mkdir build

# Compile the source code
cmake -S . -B build

# Build the executable
cmake --build build

# Run the executable
./build/BFGS
