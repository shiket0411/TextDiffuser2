#!/bin/sh

#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -g gi44
#PJM -j

module load aquarius
module load python/3.8.12
source textdiffuser2/bin/activate

python3 test.py
