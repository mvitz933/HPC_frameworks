#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "utils.h"
#include <omp.h>
#include "gaussseidel.hpp"


void GaussSeidel(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega) {
	int i,j,k=0;

    #pragma omp parallel default(none) shared(X_min, X_max, Y_min, Y_max, u_current, u_previous, omega) private(i, j, k)
	{
	for (k=X_min;k<X_max+1;k++){
      #pragma omp for
		for (j=Y_min;j<=k-1;j++){
			i = k - j;
			u_current[i][j]=u_previous[i][j]+(u_current[i-1][j]+u_previous[i+1][j]+u_current[i][j-1]+u_previous[i][j+1]-4*u_previous[i][j])*omega/4.0;

		}
	}
	}
    #pragma omp parallel default(none) shared(X_min, X_max, Y_min, Y_max, u_current, u_previous, omega) private(i, j, k)
	{
	for (k=X_max-1;k>=X_min;k--) {
      #pragma omp for
		for (j=Y_min;j<=k-1;j++) {
			i = k - j;
			u_current[X_max-j][Y_max-i]=u_previous[X_max-j][Y_max-i]+(u_current[X_max-j-1][Y_max-i]+u_previous[X_max-j+1][Y_max-i]+u_current[X_max-j][Y_max-i-1]+u_previous[X_max-j][Y_max-i+1]-4*u_previous[X_max-j][Y_max-i])*omega/4.0;
		}
	}
	}
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "utils.h"

void RedSOR(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega) {
	int i,j;
    #pragma omp parallel default(none) shared(X_min, X_max, Y_min, Y_max, u_current, u_previous, omega) private(i, j)
	{	
      #pragma omp for collapse(2)
	for (i=X_min;i<X_max;i++)
		for (j=Y_min;j<Y_max;j++)
			if ((i+j)%2==0) 
				u_current[i][j]=u_previous[i][j]+(omega/4.0)*(u_previous[i-1][j]+u_previous[i+1][j]+u_previous[i][j-1]+u_previous[i][j+1]-4*u_previous[i][j]);		         
	}
}

void BlackSOR(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega) {
	int i,j;
    #pragma omp parallel default(none) shared(X_min, X_max, Y_min, Y_max, u_current, u_previous, omega) private(i, j)
	{	
      #pragma omp for collapse(2)
	for (i=X_min;i<X_max;i++)
		for (j=Y_min;j<Y_max;j++)
			if ((i+j)%2==1) 
				u_current[i][j]=u_previous[i][j]+(omega/4.0)*(u_current[i-1][j]+u_current[i+1][j]+u_current[i][j-1]+u_current[i][j+1]-4*u_previous[i][j]); 
	}
}
