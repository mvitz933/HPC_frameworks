#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include <mpi.h>


#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

using namespace std;


int main(int argc, char *argv[]) {
	int rank, npes;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int npoints=100000;//atoi(argv[1]);
  int i;
  double x=0,y=0;
  srand48(time(NULL));

  double pi=0;
  double square=0;
  for (i = 1; i <= npoints; i++) {
    x=drand48(); y=drand48();
//    printf("random value = %g, \n", r );
    if((x*x+y*y)<1)pi+=1;
    square+=1;
  }
  pi*=4.0/npoints;

  double total;
  MPI_Reduce(&pi,&total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  if (rank==0) {
  double avg=total/npoints;
  cout<<"Nsteps="<<npoints<<","<<"uncertainty="<<sqrt((double)npoints)<<","<<"average="<<avg<<"\n";
}

  MPI_Finalize();

  printf("pi value = %g, \n", pi ); 
  return 0;
}
