#!/bin/bash

module use ~/bench/modules
module load my-dealii/8.5.0

cd build
cmake ..
make
