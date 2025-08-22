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

    
    int number_particles = n_pop;
    // int elements_per_process = number_particles / world_size;
    int *elements_per_process = new int[world_size];
    for (int i = 0; i < world_size; i++) {
        elements_per_process[i] = 0;
    }
    int i = 0;
    int k = number_particles;
    while (k > 0){
        elements_per_process[i] += 1;
        k--;
        i++;
        if (i >= world_size) {
            i = 0;
        }
    }

    srand(seed);

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

    //velocity parameters
    double initialWeight = iW, finalWeight = fW; //double initialWeight = 0.9, finalWeight = 0.4;
    double weight;
    double cognition_parameter = cogP; //0.5
    double social_parameter = socP;   // 0.5

    if (world_rank > 0){
        for(int i = 0; i < elements_per_process[world_rank]; i++) {
            // cout << "Process: " << world_rank << " Init" << endl;
            population[i] = new double[dimension];
        }

        // for(int i=0; i < elements_per_process; i++) {
        //     MPI_Recv(
        //         population[i],
        //         dimension, 
        //         MPI_DOUBLE, 
        //         0, 
        //         0, 
        //         comm, 
        //         MPI_STATUS_IGNORE
        //     );
        // }
        double *recv_vec = new double[elements_per_process[world_rank]*dimension];
        MPI_Recv(
            recv_vec,
            elements_per_process[world_rank]*dimension, 
            MPI_DOUBLE, 
            0, 
            0, 
            comm, 
            MPI_STATUS_IGNORE
        );
        int  aux = elements_per_process[world_rank] * dimension;
        for (int i = 0; i < aux; i++) {
            int row = int(i / dimension);
            int col = i % dimension;
            population[row][col] = recv_vec[i];
            // cout << row << " " << col << " " << population[row][col] << endl;
        }

        double *local_fitness = new double[elements_per_process[world_rank]];
        for(int i = 0; i<elements_per_process[world_rank]; i++){
            local_fitness[i] = objfun(population[i]);
        }

        delete []recv_vec;

        MPI_Send(
            local_fitness, 
            elements_per_process[world_rank], 
            MPI_DOUBLE,
            0, 
            world_rank, 
            comm
        );

        delete []local_fitness;
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
            int aux = elements_per_process[dest] * dimension;
            double *send_vec = new double[aux];
            for (int i = 0; i<aux; i++) {
                int row = int(i / dimension);
                int col = i % dimension;
                send_vec[i] = population[row + (dest)*elements_per_process[dest]][col];
            }
            // for(int i = 0; i < elements_per_process; i++) {
            //     MPI_Send(
            //         population[i + (dest)*elements_per_process], 
            //         dimension, 
            //         MPI_DOUBLE, 
            //         dest, 
            //         0, 
            //         comm
            //     );
            // }
            MPI_Send(
                send_vec, 
                aux, 
                MPI_DOUBLE, 
                dest, 
                0, 
                comm
            );

            delete []send_vec;
        }        
        for(int dest=0; dest<world_size; dest++){
            double local_fitness[elements_per_process[dest]];
            if(dest == 0){
                for(int i=0; i<elements_per_process[dest]; i++){
                    fitness[i] = objfun(population[i]);
                }
            }
            else{
                MPI_Recv(
                    local_fitness, 
                    elements_per_process[dest], 
                    MPI_DOUBLE, 
                    dest, 
                    dest, 
                    comm, 
                    MPI_STATUS_IGNORE
                );
                // cout << "process" << dest << " recieved fitness" << endl;
                for(int i=0; i<elements_per_process[dest]; i++){
                    fitness[i + (dest)*elements_per_process[dest]+1] = local_fitness[i];
                }
                // cout << "Process: " << world_rank << " received fitness from process " << dest << endl;
            }
        }
        for(int i=0; i<number_particles; i++){
            best_local_fitness[i] = fitness[i];
            if(fitness[i] < best_global_fitness){
                best_global_fitness = fitness[i];
                for(int j=0; j<dimension; j++){ best_global_position[j] = population[i][j]; }
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

    if (world_rank == 0) {
        for(int dest=1; dest<world_size; dest++){               
            MPI_Send(
                &functionEvaluations, 
                1, 
                MPI_INT, 
                dest, 
                0,
                comm
            );
            
        }
    }
    else{
        MPI_Recv(
            &functionEvaluations, 
            1, 
            MPI_INT, 
            0, 
            0, 
            comm, 
            MPI_STATUS_IGNORE
        );
        // cout << "Process: " << world_rank << " received function evaluations: " << functionEvaluations << endl;
    }


    int maxEval = number_particles * 100;
    if(world_rank == 0) {
        cout << "COMECOU" <<  endl;
        cout << "best global fitness: " << best_global_fitness << endl;
        cout << "best global position: ";
        for(int j=0; j<dimension; j++){ 
            cout << best_global_position[j] << " "; 
        }
    }
    while(functionEvaluations < maxEval){
        // cout << "Process: " << world_rank << " we are in the loop" << endl;
        if (world_rank == 0) {
            // cout << "best global fitness: " << best_global_fitness << endl;

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
            }
            // for each process, send its portion of the population
            for(int dest=1; dest<world_size; dest++){
                int aux = elements_per_process[dest] * dimension;
                double *send_vec = new double[aux];
                for (int i = 0; i<aux; i++) {
                    int row = int(i / dimension);
                    int col = i % dimension;
                    send_vec[i] = population[row + (dest)*elements_per_process[dest]][col];
                }
                
                MPI_Send(
                    send_vec, 
                    aux, 
                    MPI_DOUBLE, 
                    dest, 
                    0,
                    comm
                );

                delete []send_vec;
                // for(int i = 0; i < elements_per_process[dest]; i++) {
                //     MPI_Send(
                //         population[i + (dest)*elements_per_process[dest]], 
                //         dimension, 
                //         MPI_DOUBLE, 
                //         dest, 
                //         0, 
                //         comm
                //     );
                // }
                
            }

            for(int dest=0; dest<world_size; dest++){
                double local_fitness[elements_per_process[dest]];
                if(dest == 0){
                    for(int i=0; i<elements_per_process[dest]; i++){
                        fitness[i] = objfun(population[i]);
                    }
                    // cout << "Process: " << world_rank << " fitness ok" << endl;
                }
                else{
                    MPI_Recv(
                        local_fitness, 
                        elements_per_process[dest], 
                        MPI_DOUBLE, 
                        dest, 
                        dest, 
                        comm, 
                        MPI_STATUS_IGNORE
                    );
                    for(int i=0; i<elements_per_process[dest]; i++){
                        fitness[i + (dest)*elements_per_process[dest]+1] = local_fitness[i];
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

            // cout << "Best global fitness: " << best_global_fitness << endl;
            // cout << "Process: " << world_rank <<  functionEvaluations << endl;
        }
        else {
            double *recv_vec = new double[elements_per_process[world_rank]*dimension];
            MPI_Recv(
                recv_vec,
                elements_per_process[world_rank]*dimension, 
                MPI_DOUBLE, 
                0, 
                0, 
                comm, 
                MPI_STATUS_IGNORE
            );

            int  aux = elements_per_process[world_rank] * dimension;
            for (int i = 0; i < aux; i++) {
                int row = int(i / dimension);
                int col = i % dimension;
                population[row][col] = recv_vec[i];
                // cout << row << " " << col << " " << population[row][col] << endl;
            }

            double *local_fitness = new double[elements_per_process[world_rank]];
            for(int i = 0; i<elements_per_process[world_rank]; i++){
                local_fitness[i] = objfun(population[i]);
            }

            delete []recv_vec;

            MPI_Send(
                local_fitness, 
                elements_per_process[world_rank], 
                MPI_DOUBLE,
                0, 
                world_rank, 
                comm
            );

            delete []local_fitness;
            // cout << "Process: " << world_rank << " max evaluations: " << functionEvaluations << endl;
            // for(int i=0; i < elements_per_process[world_rank]; i++) {
            //     MPI_Recv(
            //         population[i],
            //         dimension, 
            //         MPI_DOUBLE, 
            //         0, 
            //         0, 
            //         comm, 
            //         MPI_STATUS_IGNORE
            //     );
            // }
            // for(int i = 0; i<elements_per_process[world_rank]; i++){
            //     fitness[i] = objfun(population[i]);
            // }
            // // cout << "Process: " << world_rank << " fitness ok" << endl;
            // // cout << "Process: " << world_rank << " fitness ok" << endl;

            // MPI_Send(
            //     fitness, 
            //     elements_per_process[world_rank], 
            //     MPI_DOUBLE,
            //     0, 
            //     world_rank, 
            //     comm
            // );
        }

        if (world_rank == 0) {
            for(int dest=1; dest<world_size; dest++){               
                MPI_Send(
                    &functionEvaluations, 
                    1, 
                    MPI_INT, 
                    dest, 
                    0,
                    comm
                );
                
            }
        }
        else{
            MPI_Recv(
                &functionEvaluations, 
                1, 
                MPI_INT, 
                0, 
                0, 
                comm, 
                MPI_STATUS_IGNORE
            );
            // cout << "Process: " << world_rank << " received function evaluations: " << functionEvaluations << endl;
        }
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
        cout << "Process: " << world_rank << " cleaning up..." << endl;
        for(i=0; i<elements_per_process[world_rank]; i++){
            delete []population[i];
        }
        delete []population;
        delete []best_local_positions;
        delete []velocities;
        delete []best_local_fitness;
        delete []fitness;
    }

    return valor_final;
}
