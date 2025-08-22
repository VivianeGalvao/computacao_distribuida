#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <vector>
#include <omp.h>
#include "pso.h"
#include "Solution.h"

#define MAX_DELTA 0.00001
//#define INF 3.40282347E+38F
#define INF 1E+21

using namespace std;
// int functionEvaluations = 0;
int maxEval = 0;
double delta;

double generate_ramdom() {
     return ((rand())/(RAND_MAX+1.0));
}

double generate_thread_safe_random(
  mt19937& rng_engine, 
  double lower_bound, 
  double upper_bound
){
  uniform_real_distribution<double> distribution(lower_bound, upper_bound);
  return distribution(rng_engine);
}

void print_individual(Particle* individual, int dimension) {
  cout << "Position: ";
  for(int j = 0; j < dimension; j++){
    cout << individual->position[j] << " ";
  }
  cout << endl;  
  cout << "Fitness: " << individual->fitness << endl;
}

double PSO(
    double (*objfun)(double*), 
    int dimension,
    int n_pop,
    int seed, 
    double *lb, 
    double *ub, 
    double iW, 
    double fW, 
    double cogP, 
    double socP,
    int num_threads
){
        
  int functionEvaluations = 0;
  int number_particles = n_pop;

  int i, j;
  srand(seed);

  Particle **population = new Particle*[number_particles];
  Particle *best_particle = new Particle();

    //velocity parameters
  double initialWeight = iW, finalWeight = fW; //double initialWeight = 0.9, finalWeight = 0.4;
  double weight;
  double cognition_parameter = cogP; //0.5
  double social_parameter = socP;   // 0.5


  int max_threads = num_threads;
  if (max_threads == 0) max_threads = 1;

  vector<mt19937> rng_engines(max_threads);
  for (int i = 0; i < max_threads; ++i) {
      unsigned int seeds[2] = {static_cast<unsigned int>(seed), static_cast<unsigned int>(i)};
      seed_seq seed_sequence(seeds, seeds + 2);
      rng_engines[i].seed(seed_sequence);
  }
  
  for(int i = 0; i<number_particles; i++){
    population[i] = new Particle();
    population[i]->position = new double[dimension];
    population[i]->velocity = new double[dimension];
    population[i]->best_position = new double[dimension];  
  }

  // initialize a population of particles with random positions and velocities
#pragma omp parallel for num_threads(max_threads)
  for(int i = 0; i<number_particles; i++){
    
    int thread_id = omp_get_thread_num();
    mt19937& my_rng = rng_engines[thread_id];
  
    //velocity parameters
    for(int j=0; j<dimension; j++){
      if(lb != NULL && ub != NULL){
          population[i]->position[j] = generate_thread_safe_random(my_rng, lb[j], ub[j]);          
      }
      else{
          population[i]->position[j] = generate_thread_safe_random(my_rng, 0.0, 1.0);
      }
      population[i]->velocity[j] = 0.0;
      population[i]->best_position[j] = population[i]->position[j];
    //velocity paramenters
    }
    population[i]->fitness = objfun(population[i]->position);
    population[i]->best_fitness = population[i]->fitness;
    // print_individual(population[i], dimension);    
  }
  //initialize fitness values
  functionEvaluations = functionEvaluations + number_particles;

  int gbest = 0;
  best_particle->fitness = population[0]->fitness;
  best_particle->position = new double[dimension];
  for(int j=0; j<dimension; j++){ 
      best_particle->position[j] = population[0]->position[j]; 
  }
  for(int i=1; i<number_particles; i++){      
    if(population[i]->fitness < best_particle->fitness){	  
      gbest = i;
    }
  }
  best_particle->fitness = population[gbest]->fitness;
  for(j=0; j<dimension; j++){ best_particle->position[j] = population[gbest]->position[j];}
  
  int iteracoes = 0;
  int maxEval = number_particles * 100;
  cout << "COMECOU " <<  "max_threads: " << max_threads << endl;

  #pragma omp parallel num_threads(max_threads) \
    default(none) reduction(+: functionEvaluations) shared(number_particles, best_particle, population, cognition_parameter, social_parameter, initialWeight, finalWeight, maxEval, dimension, objfun, lb, ub, rng_engines) \
    private(i, j, weight, gbest, iteracoes)
  // while(functionEvaluations < maxEval){
  for (iteracoes = 0; functionEvaluations < maxEval; iteracoes++){
    bool successful = false;
    weight = initialWeight - (initialWeight - finalWeight)*((double)functionEvaluations)/((double)maxEval);

    #pragma omp for
    // #pragma omp parallel for num_threads(max_threads)
    for(int i=0; i<number_particles; i++){
      int thread_id = omp_get_thread_num();
      mt19937& my_rng = rng_engines[thread_id];

      // print_individual(population[i], dimension);
      for(int j=0; j<dimension; j++){

        double a = cognition_parameter*generate_thread_safe_random(my_rng, 0.0, 1.0);
        double b = social_parameter*generate_thread_safe_random(my_rng, 0.0, 1.0);
        // printf("a: %f, b: %f\n", a, b);
        population[i]->velocity[j] = (weight*population[i]->velocity[j]) + 
                                     (a*(population[i]->best_position[j] - population[i]->position[j])) + 
                                     (b*(best_particle->position[j] - population[i]->position[j]));
        // printf("Velocity[%d][%d] = %f\n", i, j, population[i]->velocity[j]);
        population[i]->position[j] = population[i]->position[j] + population[i]->velocity[j];
        if(lb != NULL && ub != NULL){
          if(population[i]->position[j] < lb[j]){ population[i]->position[j] = lb[j]; }
          if(population[i]->position[j] > ub[j]){ population[i]->position[j] = ub[j]; }
        }
      population[i]->fitness = objfun(population[i]->position);
      }
    }

    functionEvaluations = functionEvaluations + number_particles;

    // cout << "Function evaluations: " << functionEvaluations << endl;

    for(int i=0; i<number_particles && functionEvaluations < maxEval; i++){
      if(population[i]->fitness < population[i]->best_fitness){
        population[i]->best_fitness = population[i]->fitness;
        for(int j=0; j<dimension; j++){
            population[i]->best_position[j] = population[i]->position[j];
        }
        if(population[i]->fitness < best_particle->fitness){
            best_particle->fitness = population[i]->fitness;
            for(int j=0; j<dimension; j++){
                best_particle->position[j] = population[i]->position[j];
            }
            successful = true;
            gbest = i;
        }
      }
    }
  //   cout << "Best particle" << endl;
  //   print_individual(best_particle, dimension);
  //   cout << endl;

  }

  // for(int i=0; i<dimension; i++){
  //   cout << "Best position[" << i << "]: " << best_particle->position[i] << endl;
  // }

  double valor_final = best_particle->fitness;
  delete []best_particle->position;
  delete best_particle;
  for(int i=0; i<number_particles; i++){
      delete []population[i]->position;
      delete []population[i]->best_position;
      delete []population[i]->velocity;
      
      delete population[i];
  }
  delete []population;

  return valor_final;

  // return 0;

}
