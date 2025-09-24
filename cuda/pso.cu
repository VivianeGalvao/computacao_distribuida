#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <limits>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define INF (1e38f*1e38f)

#define SEED 42
#define DIMENSION 64
#define NUMPARTICLES 32

using namespace std;

__constant__ double pso_lower_bound;
__constant__ double pso_upper_bound;
__constant__ double pso_inertial_weight;
__constant__ double pso_cognitive_param;
__constant__ double pso_social_param;


__global__ void update_global_best(float *fitness, double *global_best_fit, int n_pop) {

    int size = n_pop;
    extern __shared__ float min_values[];

    int tid = threadIdx.x;
    if (tid < size) {
        min_values[tid] = fitness[tid];
    } else {
        min_values[tid] = INF;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            min_values[tid] = min(min_values[tid], min_values[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *global_best_fit = min_values[0] < *global_best_fit ? min_values[0] : *global_best_fit;
    }
}

__global__ void calc_fitness(
    double *population,
    double *best_pos,
    float *fitness,
    float *best_fit,
    double *global_best_fit,
    int n_pop,
    int dimension
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int particle;
    int size = n_pop * dimension;

    while (i < size){
        float val = population[i];
        particle = i / dimension;
        atomicAdd(&fitness[particle], val*val);
        i+=stride;
    }
    __syncthreads();

    i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < size){
        particle = i / dimension;
        best_pos[i] = fitness[particle] < best_fit[particle] ? population[i] : best_pos[i];
        i+=stride;
    }

    i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < size){
        particle = i / dimension;
        best_fit[particle] = fitness[particle] < best_fit[particle] ? fitness[particle] : best_fit[particle];
        i+=stride;
    }
}


__global__ void init_population(double *population, double *best_pos, int seed, int popsize, int dimension) {

    int total_dimension = popsize * dimension;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < total_dimension) {
        curandState_t state;
        curand_init(seed, i, 0, &state);

        float random_value = curand_uniform(&state);
        population[i] = (pso_upper_bound - pso_lower_bound)*random_value + pso_lower_bound;
        i += stride;
    }
}

__global__ void update_positions(
    double *population,
    double *velocity,
    double *best_pos,
    double *global_best,
    int seed,
    int n_pop,
    int dimension) {

    int total_dimension = n_pop * dimension;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < total_dimension){
        curandState_t state;
        curand_init(seed, i, 0, &state);

        float r1 = curand_uniform(&state);
        float r2 = curand_uniform(&state);

        velocity[i] = (pso_inertial_weight*velocity[i]) + \
                        (pso_cognitive_param*r1*(best_pos[i] - population[i])) + \
                        (pso_social_param*r2*(global_best[i] - population[i]));

        population[i] = population[i] + velocity[i];
        population[i] = population[i] < pso_lower_bound ? pso_lower_bound : population[i];
        population[i] = population[i] > pso_upper_bound ? pso_upper_bound : population[i];

        i += stride;
    }
}

void print_population(double* dev_population, int pop_size) {
    double *h_population = (double*)malloc(pop_size * sizeof(double));
    cudaMemcpy(
        h_population,
        dev_population,
        pop_size * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < pop_size; i++) {
        printf("population[%d] = %f\n", i, h_population[i]);
    }
    free(h_population);
}

void print_fitness(float* dev_fitness, int n_pop){
    float *h_fitness = (float*)malloc(n_pop * sizeof(float));
    cudaMemcpy(h_fitness, dev_fitness, n_pop * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_pop; i++) {
        printf("fitness[%d] = %f\n", i, h_fitness[i]);
    }
    free(h_fitness);
}

void print_best_local_fitness(double* dev_best_fit, int n_pop){
    float *h_best_fitness = (float*)malloc(n_pop * sizeof(float));
    cudaMemcpy(h_best_fitness, dev_best_fit, n_pop * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_pop; i++) {
        printf("best fitness[%d] = %f\n", i, h_best_fitness[i]);
    }
    free(h_best_fitness);
}

void print_best_position(double* dev_best_pos, int pop_size){
    double *h_best_pop = (double*)malloc(pop_size * sizeof(double));
    cudaMemcpy(
        h_best_pop,
        dev_best_pos,
        pop_size * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < pop_size; i++) {
        printf("best pop[%d] = %f\n", i, h_best_pop[i]);
    }
    free(h_best_pop);
}

int main(int argc, char **argv){

    float tempo_total_acumulado = 0.0f;
    float milliseconds = 0;

    int dimension = DIMENSION;
    int n_pop = NUMPARTICLES;
    int run_number = 1;
    int seed = SEED;

    const int pop_size = dimension * n_pop;
    int threads = DIMENSION;

    // echo "$dimension,$n_pop,$threads,$i,$TIME_STR"

    if (argc > 1) {
        // If command line arguments are provided, use them to set parameters
        dimension = atoi(argv[1]);
        n_pop = atoi(argv[2]);
        threads = atoi(argv[3]);
        run_number = atoi(argv[4]);
    }

    int blocks = (n_pop + threads - 1) / threads;

    //memoria de constantes
    double h_lb = -5, h_ub = 5, h_IW = 0.4, h_CP = 0.5, h_SP = 0.5;
    cudaMemcpyToSymbol(pso_lower_bound, &h_lb, sizeof(double));
    cudaMemcpyToSymbol(pso_upper_bound, &h_ub, sizeof(double));
    cudaMemcpyToSymbol(pso_inertial_weight, &h_IW, sizeof(double));
    cudaMemcpyToSymbol(pso_cognitive_param, &h_CP, sizeof(double));
    cudaMemcpyToSymbol(pso_social_param, &h_SP, sizeof(double));

    // population, velocity
    double *dev_population, *dev_velocity, *dev_best_pos;
    double *dev_global_best, *dev_global_best_fitness;
    cudaMalloc((void**)&dev_population, pop_size * sizeof(double));
    cudaMalloc((void**)&dev_velocity, pop_size * sizeof(double));
    cudaMalloc((void**)&dev_best_pos, pop_size * sizeof(double));
    cudaMalloc((void**)&dev_global_best, dimension * sizeof(double));
    cudaMalloc((void**)&dev_global_best_fitness, sizeof(double));

    float *dev_fitness, *dev_best_fit;
    cudaMalloc((void**)&dev_fitness, n_pop * sizeof(float));
    cudaMalloc((void**)&dev_best_fit, n_pop * sizeof(float));

    // inicializa velocidade inicial com 0
    cudaMemset(dev_velocity, 0.0f, pop_size * sizeof(double));

    // inicializa fitness com zero e best individuals fitness com infinito
    float *h_fitness_init = (float*)malloc(n_pop * sizeof(float));
    float *h_best_fitness_init = (float*)malloc(n_pop * sizeof(float));
    for (int i = 0; i < n_pop; i++) {
        h_best_fitness_init[i] = INF;
        h_fitness_init[i] = 0.0f;
    }
    cudaMemcpy(
        dev_best_fit,
        h_best_fitness_init,
        n_pop * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        dev_fitness,
        h_fitness_init,
        n_pop * sizeof(float),
        cudaMemcpyHostToDevice
    );
    free(h_fitness_init);
    free(h_best_fitness_init);

    double h_inf = INF;
    cudaMemcpy(
        dev_global_best_fitness,
        &h_inf,
        sizeof(double),
        cudaMemcpyHostToDevice
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    init_population<<<blocks, threads>>>(
        dev_population,
        dev_best_pos,
        seed,
        n_pop,
        dimension
    );

    calc_fitness<<<blocks, threads, n_pop * sizeof(double)>>>(
        dev_population,
        dev_best_pos,
        dev_fitness,
        dev_best_fit,
        dev_global_best_fitness,
        n_pop,
        dimension
    );

    update_global_best<<<1, n_pop, n_pop * sizeof(float)>>>(
        dev_fitness,
        dev_global_best_fitness,
        n_pop
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    tempo_total_acumulado += milliseconds;

    double h_global_best_fitness;
    // cudaMemcpy(&h_global_best_fitness, dev_global_best_fitness, sizeof(double), cudaMemcpyDeviceToHost);
    // printf("Global best fitness: %f\n", h_global_best_fitness);

    int generations = 10;
    int iter = 0;
    cudaEventRecord(start);
    while (iter < generations) {
        update_positions<<<blocks, threads>>>(
            dev_population,
            dev_velocity,
            dev_best_pos,
            dev_global_best,
            seed+iter,
            n_pop,
            dimension
        );

        cudaMemset(dev_fitness, 0.0f, n_pop * sizeof(float));
        calc_fitness<<<blocks, threads, n_pop * sizeof(double)>>>(
            dev_population,
            dev_best_pos,
            dev_fitness,
            dev_best_fit,
            dev_global_best_fitness,
            n_pop,
            dimension
        );

        update_global_best<<<1, n_pop, n_pop * sizeof(float)>>>(
            dev_fitness,
            dev_global_best_fitness,
            n_pop
        );
        iter++;
        // printf("Iteracao %d\n", iter);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    tempo_total_acumulado += milliseconds;

    cudaMemcpy(&h_global_best_fitness, dev_global_best_fitness, sizeof(double), cudaMemcpyDeviceToHost);
    // printf("Global best fitness: %f\n", h_global_best_fitness);
    // printf("Tempo total (ms): %f\n", tempo_total_acumulado);
    // "dimension,n_pop,n_thread,n_block,run_number,execution_time":
    printf("%d,%d,%d,%d,%d,%f,%.5f\n", dimension, n_pop, threads, blocks, run_number, h_global_best_fitness,tempo_total_acumulado);


    cudaFree(dev_population);
    cudaFree(dev_velocity);
    cudaFree(dev_best_pos);
    cudaFree(dev_best_fit);
    cudaFree(dev_fitness);
    cudaFree(dev_global_best);

    return 0;

}