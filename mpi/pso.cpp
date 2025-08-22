#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <mpi.h>
#include "pso.h"
#include "Solution.h"


#define MAX_DELTA 0.00001
//#define INF 3.40282347E+38F
#define INF 1E+21

using namespace std;
int functionEvaluations = 0;
int maxEval = 0;
double delta;

double generate_ramdom() {
    return ((rand())/(RAND_MAX+1.0));
}

double PSO(
    double (*objfun)(double*), // Renamed to avoid conflict with global objfun
    int dimension,
    int n_pop,
    int seed,
    double *lb,
    double *ub,
    double iW,
    double fW,
    double cogP,
    double socP,
    int world_rank,     // MPI rank of this process
    int world_size,     // Total number of MPI processes
    MPI_Comm comm       // MPI communicator
){
    if (world_rank == 0) {
        cout << "Starting PSOzinho..." << endl;
    }


    int *elements_per_process = new int[world_size];
    for (int i = 0; i < world_size; i++) {
        elements_per_process[i] = 0;
    }
    int i = 0;
    int k = n_pop;
    while (k > 0){
        elements_per_process[i] += 1;
        k--;
        i++;
        if (i >= world_size) {
            i = 0;
        }
    }

    srand(seed+world_rank);

    int number_particles = elements_per_process[world_rank];

    double **population = new double*[number_particles];
    double **best_local_positions = new double*[number_particles];
    double **velocities = new double*[number_particles];

    double *best_local_fitness = new double[number_particles];
    double *fitness = new double[number_particles];
    for(int i = 0; i < number_particles; i++) {
        fitness[i] = INF; // Initialize fitness to a large value
    }

    double *best_global_position = new double[dimension];
    double best_global_fitness = INF;

    double *recv_best_fitness = new double[world_size];

    //velocity parameters
    double initialWeight = iW, finalWeight = fW; //double initialWeight = 0.9, finalWeight = 0.4;
    double weight;
    double cognition_parameter = cogP; //0.5
    double social_parameter = socP;   // 0.5


    cout << "Initializing population..." << endl;
    for(int i = 0; i<number_particles; i++){
        population[i] = new double[dimension];
        velocities[i] = new double[dimension];
        best_local_positions[i] = new double[dimension];
        //velocity parameters
        for(int j=0; j<dimension; j++){

            if(lb != NULL && ub != NULL){
                population[i][j] = (ub[j]-lb[j])*generate_ramdom() + lb[j];
                velocities[i][j] = 0.0;
            }
            else{
                population[i][j] = generate_ramdom();
                velocities[i][j] = 0.0;
            }
            if(world_rank == 0){
                cout << "Particle " << i << " Position[" << j << "]: " << population[i][j] << endl;
            } 
            best_local_positions[i][j] = population[i][j];
        }

        fitness[i] = objfun(population[i]);
        functionEvaluations += world_size;
        best_local_fitness[i] = fitness[i];
        if(fitness[i] < best_global_fitness){
            best_global_fitness = fitness[i];
            for(int j=0; j<dimension; j++){ best_global_position[j] = population[i][j]; }
        }
    }

    maxEval = number_particles * 100;
    if(world_rank == 0) {
        cout << "COMECOU" <<  endl;
        cout << "best global fitness: " << best_global_fitness << endl;
        // cout << "best global position: ";
        // for(int j=0; j<dimension; j++){ 
        //     cout << best_global_position[j] << " "; 
        // }
    }
    while(functionEvaluations < maxEval){
        cout << "Process: " << world_rank << " " << best_global_fitness << endl;

        weight = initialWeight - (initialWeight - finalWeight)*((double)functionEvaluations)/((double)maxEval);

        for(int i=0; i<number_particles && functionEvaluations < maxEval; i++){
            for(int j=0; j<dimension; j++){	    
                velocities[i][j] = (weight*velocities[i][j]) + 
                                    (cognition_parameter*generate_ramdom()*(best_local_positions[i][j] - population[i][j])) + 
                                    (social_parameter*generate_ramdom()*(best_global_position[j] - population[i][j]));


                population[i][j] = population[i][j] + velocities[i][j];
                if(lb != NULL && ub != NULL){
                    if(population[i][j] < lb[j]){ population[i][j] = lb[j]; }
                    if(population[i][j] > ub[j]){ population[i][j] = ub[j]; }
                }
            }
            fitness[i] = objfun(population[i]);              
    	    functionEvaluations += world_size;
        
            for(int j=0; j<dimension; j++){
                if(fitness[i] < best_local_fitness[i]){
                    best_local_fitness[i] = fitness[i];
                    for(j=0; j<dimension; j++){
                        best_local_positions[i][j] = population[i][j];
                    }
                    if(fitness[i] < best_global_fitness){
                        best_global_fitness = fitness[i];
                        for(j=0; j<dimension; j++){
                            best_global_position[j] = population[i][j];
                        }
                    }
                }
            }
        }
    }



    double valor_final = 0;

    MPI_Reduce(
        &best_global_fitness, 
        &valor_final, 
        1, 
        MPI_DOUBLE, 
        MPI_MIN, 
        0, 
        comm
    );

    for(i=0; i<number_particles; i++){
        delete []population[i];
        delete []best_local_positions[i];
        delete []velocities[i];
    }
    delete []population;
    delete []best_local_positions;
    delete []velocities;
    delete []best_local_fitness;
    delete []fitness;
    delete []best_global_position;

    return valor_final;

}
