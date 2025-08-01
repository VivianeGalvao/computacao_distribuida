#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <mpi.h>
#include "pso.h"
#include "Solution.h"

#define NUMPARTICLES 9
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

    functionEvaluations = 0;
    int number_particles = NUMPARTICLES;

    int i, j;
    srand(seed + world_rank);

    Particle **population = new Particle*[NUMPARTICLES];
    Particle *best_particle = new Particle();

    int elements_per_process = number_particles / world_size;
    //velocity parameters
    double initialWeight = iW, finalWeight = fW; //double initialWeight = 0.9, finalWeight = 0.4;
    double weight;
    double cognition_parameter = cogP; //0.5
    double social_parameter = socP;   // 0.5

    // //initialize a population of particles with random positions and velocities
    if (world_rank == 0) {
        cout << "Initializing population..." << endl;
        for(int i = 0; i<number_particles; i++){
            population[i] = new Particle();
            population[i]->position = new double[dimension];
            population[i]->velocity = new double[dimension];
            population[i]->best_position = new double[dimension];
            //velocity parameters
            for(int j=0; j<dimension; j++){

                if(lb != NULL && ub != NULL){
                    population[i]->position[j] = (ub[j]-lb[j])*generate_ramdom() + lb[j];
                    //population[i]->velocity[j] = (ub[j]-lb[j])*generate_ramdom() + lb[j];
                    population[i]->velocity[j] = 0.0;
                }
                else{
                    population[i]->position[j] = generate_ramdom();
                    population[i]->velocity[j] = 0.0;
                }
                population[i]->best_position[j] = population[i]->position[j];
            //velocity paramenters
            }
            population[i]->best_fitness = population[i]->fitness;
        }
    }
    //population intialized

    functionEvaluations = 0;
    if (world_rank == 0) {
        for(int i = 0; i<number_particles; i++){
            population[i]->fitness = objfun(population[i]->position);
            functionEvaluations++;
        }
        cout << "Total function evaluations: " << functionEvaluations << endl;
    }

    int gbest = 0;
    if (world_rank == 0) {
        best_particle->fitness = population[0]->fitness;
        best_particle->position = new double[dimension];
        for(j=0; j<dimension; j++){ 
            best_particle->position[j] = population[0]->position[j]; 
        }
        for(i=1; i<number_particles; i++){      
            if(population[i]->fitness < best_particle->fitness){	  
            gbest = i;
            }
        }
        best_particle->fitness = population[gbest]->fitness;
        for(j=0; j<dimension; j++){ best_particle->position[j] = population[gbest]->position[j];}
    }
    
    int iteracoes = 0, maxEval = 10;
    cout << "COMECOU" <<  endl;
    while(functionEvaluations < maxEval){
        bool successful = false;
        weight = initialWeight - (initialWeight - finalWeight)*((double)functionEvaluations)/((double)maxEval);

        for(i=0; i<number_particles && functionEvaluations < maxEval; i++){
        for(j=0; j<dimension; j++){	    
                population[i]->velocity[j] = (weight*population[i]->velocity[j]) + (cognition_parameter*generate_ramdom()*(population[i]->best_position[j] -
                            population[i]->position[j])) + (social_parameter*generate_ramdom()*(best_particle->position[j] - population[i]->position[j]));


                population[i]->position[j] = population[i]->position[j] + population[i]->velocity[j];
                if(lb != NULL && ub != NULL){
                if(population[i]->position[j] < lb[j]){ population[i]->position[j] = lb[j]; }
                if(population[i]->position[j] > ub[j]){ population[i]->position[j] = ub[j]; }
                }
            }
            population[i]->fitness = objfun(population[i]->position);
            functionEvaluations++;
            if(population[i]->fitness < population[i]->best_fitness){
                population[i]->best_fitness = population[i]->fitness;
                for(j=0; j<dimension; j++){
                    population[i]->best_position[j] = population[i]->position[j];
                }
                if(population[i]->fitness < best_particle->fitness){
                    best_particle->fitness = population[i]->fitness;
                    for(j=0; j<dimension; j++){
                        best_particle->position[j] = population[i]->position[j];
                    }
                    successful = true;
                    gbest = i;
                }
        }
        }    
    }

    double valor_final = best_particle->fitness;
    delete []best_particle->position;
    delete best_particle;
    for(i=0; i<number_particles; i++){
        delete []population[i]->position;
        delete []population[i]->best_position;
        delete []population[i]->velocity;
        delete population[i];
    }
    delete []population;

    return valor_final;
    // return -1;

    }
