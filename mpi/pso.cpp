#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <mpi.h>
#include "pso.h"
#include "Solution.h"

#define NUMPARTICLES 500
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

    
    int number_particles = NUMPARTICLES;
    int elements_per_process = number_particles / world_size;

    int i, j;
    srand(seed);

    double **population = new double*[number_particles];
    double **best_local_positions = new double*[number_particles];
    double **velocities = new double*[number_particles];

    double *best_local_fitness = new double[number_particles];
    double *fitness = new double[number_particles];

    double *best_global_position = new double[dimension];
    double best_global_fitness = INF;

    //velocity parameters
    double initialWeight = iW, finalWeight = fW; //double initialWeight = 0.9, finalWeight = 0.4;
    double weight;
    double cognition_parameter = cogP; //0.5
    double social_parameter = socP;   // 0.5

    if (world_rank > 0){
        for(int i = 0; i < elements_per_process; i++) {
            // cout << "Process: " << world_rank << " Init" << endl;
            population[i] = new double[dimension];
        }
        for(int i=0; i < elements_per_process; i++) {
            MPI_Recv(
                population[i],
                dimension, 
                MPI_DOUBLE, 
                0, 
                0, 
                comm, 
                MPI_STATUS_IGNORE
            );
        }
        for(int i = 0; i<elements_per_process; i++){
            fitness[i] = objfun(population[i]);
        }
        // cout << "Process: " << world_rank << " fitness ok" << endl;

        MPI_Send(
            fitness, 
            elements_per_process, 
            MPI_DOUBLE,
            0, 
            world_rank, 
            comm
        );
    }
    else {
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
                best_local_positions[i][j] = population[i][j];
            }
        }
        
        // for each process, send its portion of the population
        for(int dest=1; dest<world_size; dest++){    
            for(int i = 0; i < elements_per_process; i++) {
                MPI_Send(
                    population[i + (dest)*elements_per_process], 
                    dimension, 
                    MPI_DOUBLE, 
                    dest, 
                    0, 
                    comm
                );
            }
              
        }        
        for(int dest=0; dest<world_size; dest++){
            double local_fitness[elements_per_process];
            if(dest == 0){
                for(int i=0; i<elements_per_process; i++){
                    fitness[i] = objfun(population[i]);
                }
                // cout << "Process: " << world_rank << " fitness ok" << endl;
            }
            else{
                MPI_Recv(
                    local_fitness, 
                    elements_per_process, 
                    MPI_DOUBLE, 
                    dest, 
                    dest, 
                    comm, 
                    MPI_STATUS_IGNORE
                );
                for(int i=0; i<elements_per_process; i++){
                    fitness[i + (dest)*elements_per_process] = local_fitness[i];
                }
                // cout << "Process: " << world_rank << " received fitness from process " << dest << endl;
            }
        }
        for(int i=0; i<number_particles; i++){
            best_local_fitness[i] = fitness[i];
            if(fitness[i] < best_global_fitness){
                best_global_fitness = fitness[i];
                for(j=0; j<dimension; j++){ best_global_position[j] = population[i][j]; }
            }
        }
        // cout << "Fitness vector: ";
        // for(int i = 0; i < number_particles; i++) {
        //     cout << fitness[i] << " ";
        // }
        // cout << endl;
        
        functionEvaluations += number_particles;

        // cout << "population ready..." << endl;
    }

    MPI_Barrier(comm);

    int maxEval = dimension * 10;
    if(world_rank == 0) {
        cout << "COMECOU" <<  endl;
    }
    while(functionEvaluations < maxEval){
        // cout << "Process: " << world_rank << " we are in the loop" << endl;
        if (world_rank == 0) {
            weight = initialWeight - (initialWeight - finalWeight)*((double)functionEvaluations)/((double)maxEval);

            for(i=0; i<number_particles && functionEvaluations < maxEval; i++){
                for(j=0; j<dimension; j++){	    
                    velocities[i][j] = (weight*velocities[i][j]) + 
                                       (cognition_parameter*generate_ramdom()*(best_local_positions[i][j] - population[i][j])) + 
                                       (social_parameter*generate_ramdom()*(best_global_position[j] - population[i][j]));


                    population[i][j] = population[i][j] + velocities[i][j];
                    if(lb != NULL && ub != NULL){
                        if(population[i][j] < lb[j]){ population[i][j] = lb[j]; }
                        if(population[i][j] > ub[j]){ population[i][j] = ub[j]; }
                    }
                }               
            }
            // for each process, send its portion of the population
            for(int dest=1; dest<world_size; dest++){    
                for(int i = 0; i < elements_per_process; i++) {
                    MPI_Send(
                        population[i + (dest)*elements_per_process], 
                        dimension, 
                        MPI_DOUBLE, 
                        dest, 
                        0, 
                        comm
                    );
                }
                
            }

            for(int dest=0; dest<world_size; dest++){
                double local_fitness[elements_per_process];
                if(dest == 0){
                    for(int i=0; i<elements_per_process; i++){
                        fitness[i] = objfun(population[i]);
                    }
                    // cout << "Process: " << world_rank << " fitness ok" << endl;
                }
                else{
                    MPI_Recv(
                        local_fitness, 
                        elements_per_process, 
                        MPI_DOUBLE, 
                        dest, 
                        dest, 
                        comm, 
                        MPI_STATUS_IGNORE
                    );
                    for(int i=0; i<elements_per_process; i++){
                        fitness[i + (dest)*elements_per_process] = local_fitness[i];
                    }
                    // cout << "Process: " << world_rank << " received fitness from process " << dest << endl;
                }
            }
            // cout << "Fitness vector: ";
            // for(int i = 0; i < number_particles; i++) {
            //     cout << fitness[i] << " ";
            // }
            // cout << endl;
            for(int i=0; i<number_particles; i++){
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
            functionEvaluations += number_particles;
            cout << "Best global fitness: " << best_global_fitness << endl;
            cout << "Process: " << world_rank <<  functionEvaluations << endl;
        }
        else {
            for(int i=0; i < elements_per_process; i++) {
                MPI_Recv(
                    population[i],
                    dimension, 
                    MPI_DOUBLE, 
                    0, 
                    0, 
                    comm, 
                    MPI_STATUS_IGNORE
                );
            }
            for(int i = 0; i<elements_per_process; i++){
                fitness[i] = objfun(population[i]);
            }
            // cout << "Process: " << world_rank << " fitness ok" << endl;

            MPI_Send(
                fitness, 
                elements_per_process, 
                MPI_DOUBLE,
                0, 
                world_rank, 
                comm
            );
        }

        MPI_Bcast(
                &functionEvaluations,
                1,
                MPI_DOUBLE, 
                0,
                comm
            );

        MPI_Barrier(comm);   
    }

    double valor_final = best_global_fitness;

    if (world_rank == 0) {
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
    } 
    else {
        // cout << "Process: " << world_rank << " cleaning up..." << endl;
        for(i=0; i<elements_per_process; i++){
            delete []population[i];
        }
        delete []population;
        delete []best_local_positions;
        delete []velocities;
        delete []best_local_fitness;
        delete []fitness;
    }

    MPI_Barrier(comm); 

    return valor_final;
}
