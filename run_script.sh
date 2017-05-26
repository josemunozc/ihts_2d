#!/bin/bash

CODE=mycode
      
mpirun -np $1 ./$CODE input.prm > output.txt

mv output/*.vtu visualization
mv output/*.txt summary_output
mv output/* preheatings
