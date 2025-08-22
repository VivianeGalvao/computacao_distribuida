#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>

#include "string.h"

#include "pso.h"

#define INF 1E+21
#define asl cur_ASL

#define SEED 42
#define DIMENSION 30
#define NUMPARTICLES 10


using namespace std;


int global_dimension = DIMENSION;
double objfun(double *x){
    // Example objective function: Sphere function
    double sum = 0.0;
    for(int i = 0; i < global_dimension; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

// double objfun(double *x) {
//     double sum = 0.0;
//     for(int i=0; i < global_dimension-1; i++) {
//         double sum_1 = (x[i]*x[i] - x[i+1]);
//         double sum_2 = (x[i] - 1);
//         sum += 100*sum_1*sum_1 + sum_2*sum_2;
//     }
//     return sum;
// }

int main(int argc, char** argv)
{   

    int dimension = DIMENSION;
    int n_pop = NUMPARTICLES;
    int seed = SEED;
    int n_threads = 2;

    if (argc > 1) {
        // If command line arguments are provided, use them to set parameters
        dimension = atoi(argv[1]);
        n_pop = atoi(argv[2]);
        n_threads = atoi(argv[3]);
    }
    global_dimension = dimension; // Set global dimension for objfun

    cout << "Starting PSO..." << endl;
    double *lb = new double[DIMENSION];
    double *ub = new double[DIMENSION];
    double IW = 0.9; // Initial weight
    double FW = 0.4; // Final weight
    double CP = 0.5; // Cognitive parameter
    double SP = 0.5; // Social parameter

    for(int i = 0; i < DIMENSION; i++) {
        lb[i] = -5.0; // Lower bound
        ub[i] = 5.0;  // Upper bound
    }
    cout << "Dimension: " << dimension << endl;
    cout << "Population size: " << n_pop << endl;
    cout << "Number of threads: " << n_threads << endl;
    
    double result = PSO(
        &objfun, 
        dimension,
        n_pop, 
        seed,
        lb, 
        ub, 
        IW, 
        FW, 
        CP, 
        SP,
        n_threads
    );
    cout << "Best fitness: " << result << endl;
    return result;
}
