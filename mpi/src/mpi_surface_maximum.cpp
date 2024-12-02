// This program is to caculate PI using MPI
// The algorithm is based on Monte Carlo method. The Monte Carlo method randomly picks up a large number of points in a square. It only counts the ratio of pints in side the circule.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include "surfaces.hpp"

#define N 1E9
#define d 1E-9

int main (int argc, char* argv[])
{
    int rank, size, error, i, result=0, sum=0;
    double pi=0.0, begin=0.0, end=0.0, x, y, a, b, z1, z2, z;
    double z_max_local=-10E-10, z_max_global=-10E-10; 
    error=MPI_Init (&argc, &argv);
    
    //Get process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    //Get processes Number
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    //Synchronize all processes and get the begin time
    MPI_Barrier(MPI_COMM_WORLD);
    begin = MPI_Wtime();
    
    srand((int)time(0));
    
    //Each process will caculate a part of the sum
    for (i=rank; i<N; i+=size)
    {
        x=20*PI*(rand()/(RAND_MAX+1.0)-0.5);
        y=20*PI*(rand()/(RAND_MAX+1.0)-0.5);

        a=calculate_a(x);
        b=calculate_b(y); 
        z1=calculate_z1(x,y);
        z2=calculate_z2(a,b);
        z=z1+z2;

        if(z>z_max_local)
            z_max_local=z;
    }
    
    //Sum up all results
    MPI_Reduce(&z_max_local, &z_max_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    //Synchronize all processes and get the end time
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    //Caculate and print PI
    if (rank==0)
    {
        pi=4.0*sum/N;
        printf("np=%2d;    Time=%fs;    max=%0.9f\n", size, end-begin, z_max_global);
    }
    
    error=MPI_Finalize();
    
    return 0;
}
