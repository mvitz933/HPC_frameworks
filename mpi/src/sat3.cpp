#include <mpi.h>
#include <stdio.h>
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)
#define FACTOR 2048
int check_circuit(int id, int z){
int v[16];   /*each element is bit of z*/
int i;

for (i=0; i<16; i++){
v[i]=EXTRACT_BIT(z,i);
}

if( (v[0]  || v[1])   && (!v[1] || !v[3]) && (v[2] || v[3])
 && (!v[3] || !v[4])  && (v[4]  || !v[5]) 
 && (v[5]  || !v[6])  && (v[5]  || v[6])
 && (v[6]  || !v[15]) && (v[7]  || !v[8])
 && (!v[7] || !v[13]) && (v[8]  || v[9])
 && (v[8]  || !v[9])  && (!v[9] || !v[10])
 && (v[9]  || v[11])  && (v[10] || v[11])
 && (v[12] || v[13])  && (v[13] || !v[14])
 && (v[14] || v[15])){

printf( "\n%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d" , id, v[0], v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]);
fflush(stdout);
return 1;
}
return 0; 
}

int main(int argc, char *argv[]){
int global_solutions, solutions, i, id, p;
int check_circuit(int, int);
double elapsed_time;

MPI_Init(&argc,&argv);
MPI_Barrier(MPI_COMM_WORLD);
elapsed_time=-MPI_Wtime();
MPI_Comm_rank(MPI_COMM_WORLD, &id);
MPI_Comm_size(MPI_COMM_WORLD, &p);

solutions=0;
for(i=id;i<65536*FACTOR; i+=p){
  solutions+=check_circuit(id,i);
}

MPI_Reduce(&solutions, &global_solutions, 1, 
	    MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
elapsed_time+=MPI_Wtime();
printf("\nProcess %d is done.\n	", id);
fflush(stdout);
MPI_Finalize();

if(id==0){
printf("There are  %d global solutions.\n Time taken (with a factor of %d) was %f\n", 
global_solutions/FACTOR,FACTOR,elapsed_time);
}

return 0;
}
