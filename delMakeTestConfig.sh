#!/bin/bash

# this prevents it from crashing if build directory doesn't exist
rm -r build || true
mkdir build
cd build
../config.sh
make -j8
make test
