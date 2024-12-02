#!/bin/bash

# this is an example of running the mpi command on a local system in which hcoll is not enabled

mpiexec --mca coll_hcoll_enable 0 --mca pml ob1 --mca btl ^openib -n 8 ./build/bin/mpi_pi_trapezoid2
