#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>

#include "string.h"

#include "pso.h"

#define INF 1E+21
#define asl cur_ASL

#define SEED 6

using namespace std;


double objfun(double *x){
    // Example objective function: Sphere function
    double sum = 0.0;
    for(int i = 0; i < 2; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

int main(int argc, char** argv)
{   
    cout << "Starting PSO..." << endl;
    double *lb = NULL;
    double *ub = NULL;
    double IW = 0.9; // Initial weight
    double FW = 0.4; // Final weight
    double CP = 0.5; // Cognitive parameter
    double SP = 0.5; // Social parameter
    double result = PSO(&objfun, 1000, SEED, lb, ub, IW, FW, CP, SP);
    cout << "Best fitness: " << result << endl;
    return result;
}
