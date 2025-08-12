#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <mpi.h>

#include "string.h"

#include "pso.h"

#define INF 1E+21
#define asl cur_ASL

#define SEED 6
#define DIMENSION 10000

using namespace std;


// double objfun(double *x){
//     // Example objective function: Sphere function
//     double sum = 0.0;
//     for(int i = 0; i < DIMENSION; i++) {
//         sum += x[i] * x[i];
//     }
//     return sum;
// }

double objfun(double *x) {
    double sum = 0.0;
    for(int i=0; i < DIMENSION-1; i++) {
        double sum_1 = (x[i]*x[i] - x[i+1]);
        double sum_2 = (x[i] - 1);
        sum += 100*sum_1*sum_1 + sum_2*sum_2;
    }
    return sum;
}

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        std::cout << "Starting PSO..." << std::endl;
    }
    double *lb = new double[DIMENSION];
    double *ub = new double[DIMENSION];

    for(int i = 0; i < DIMENSION; i++) {
        lb[i] = -100.0; // Lower bound
        ub[i] = 100.0;  // Upper bound
    }

    int dimension = DIMENSION;
    int seed = SEED;
    double IW = 0.9; // Initial weight
    double FW = 0.4; // Final weight
    double CP = 0.5; // Cognitive parameter
    double SP = 0.5; // Social parameter

    double result = PSO(
        &objfun,
        dimension,
        seed,
        lb,
        ub,
        IW,
        FW,
        CP,
        SP,
        world_rank,
        world_size,
        MPI_COMM_WORLD // Pass the communicator
    );

    if (world_rank == 0) {
        cout << "Final result: " << result << endl;
    }

    MPI_Finalize();
    return 0;
}
