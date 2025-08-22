#ifndef ESTRATEGIAS_H_INCLUDED
#define ESTRATEGIAS_H_INCLUDED
#include "Solution.h"


double PSO(double (*objfun)(double*), 
    int dimension,
    int n_pop, // Number of particles
    int seed, double *lb, 
    double *ub, 
    double iW, 
    double fW, 
    double cogP, 
    double socP,
    int world_rank,     // Added MPI parameters
    int world_size,     // Added MPI parameters
    MPI_Comm comm       // Added MPI parameters
);




#endif // ESTRATEGIAS_H_INCLUDED
