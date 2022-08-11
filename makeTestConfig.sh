#!/bin/bash

cd build
../config.sh
make -j8
make test
