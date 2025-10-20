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
#define DIMENSION 8192
#define NUMPARTICLES 512
#define NTHREADS 512

using namespace std;

__constant__ float pso_lower_bound;
__constant__ float pso_upper_bound;
__constant__ float pso_inertial_weight;
__constant__ float pso_cognitive_param;
__constant__ float pso_social_param;


__global__ void find_global_best_and_update_pos_warp(
    float* best_fit,
    float* best_pos,
    float* global_best_fit,
    float* global_best_pos,
    int n_pop,
    int dimension
) {

    extern __shared__ char s_mem[];
    unsigned int warp_size = 32;
    unsigned int warps_in_block = blockDim.x / warp_size;
    if (blockDim.x % warp_size != 0) warps_in_block++;

    float* warp_min_results = (float*)s_mem;
    int* warp_idx_results = (int*)(warp_min_results + warps_in_block);

    int tid_local = threadIdx.x;
    unsigned int lane_id = tid_local % warp_size;
    unsigned int warp_id = tid_local / warp_size;

    float my_val = (tid_local < n_pop) ? best_fit[tid_local] : INF;
    int my_idx = tid_local;

    float warp_min_val = my_val;
    int warp_min_idx = my_idx;

    unsigned int active_mask = __activemask();
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        float neighbour_val = __shfl_down_sync(active_mask, warp_min_val, offset, warp_size);
        int neighbour_idx = __shfl_down_sync(active_mask, warp_min_idx, offset, warp_size);

        if (neighbour_val < warp_min_val) {
            warp_min_val = neighbour_val;
            warp_min_idx = neighbour_idx;
        }

        else if (neighbour_val == warp_min_val && neighbour_idx < warp_min_idx) {
             warp_min_idx = neighbour_idx;
        }
    }

    if (lane_id == 0) {
        warp_min_results[warp_id] = warp_min_val;
        warp_idx_results[warp_id] = warp_min_idx;
    }

    __syncthreads();

    if (tid_local == 0) {
        float block_min_val = INF;
        int block_min_idx = -1;

        for (unsigned int i = 0; i < warps_in_block; ++i) {
             bool warp_processed_data = (i * warp_size < n_pop);
             if (warp_processed_data && warp_min_results[i] < block_min_val) {
                block_min_val = warp_min_results[i];
                block_min_idx = warp_idx_results[i];
            }
            else if (warp_processed_data && warp_min_results[i] == block_min_val && warp_idx_results[i] < block_min_idx) {
                 block_min_idx = warp_idx_results[i];
            }
        }

        if (block_min_idx != -1 && block_min_val < *global_best_fit) {
            *global_best_fit = block_min_val;
            int best_particle_id = block_min_idx;
            for (int d = 0; d < dimension; ++d) {
                global_best_pos[d] = best_pos[best_particle_id * dimension + d];
            }
        }
    }
}

__global__ void find_global_best_and_update_pos(
    float* best_fit,
    float* best_pos,
    float* global_best_fit,
    float* global_best_pos,
    int n_pop,
    int dimension
) {
    extern __shared__ char s_mem[];
    float* min_values = (float*)s_mem;
    int* argmin_values = (int*)(min_values + blockDim.x);

    int tid = threadIdx.x;

    if (tid < n_pop) {
        min_values[tid] = best_fit[tid];
        argmin_values[tid] = tid;
    } else {
        min_values[tid] = INF;
        argmin_values[tid] = -1;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (min_values[tid + stride] < min_values[tid]) {
                min_values[tid] = min_values[tid + stride];
                argmin_values[tid] = argmin_values[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (min_values[0] < *global_best_fit) {
            *global_best_fit = min_values[0];
            int best_particle_id = argmin_values[0];
            for (int d = 0; d < dimension; ++d) {
                global_best_pos[d] = best_pos[best_particle_id * dimension + d];
            }
        }
    }
}

__global__ void update_global_best(float *fitness, float *global_best_fit, int n_pop) {

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
            min_values[tid] = min_values[tid] < min_values[tid + stride] ? min_values[tid] : min_values[tid + stride];   //min(min_values[tid], min_values[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *global_best_fit = min_values[0] < *global_best_fit ? min_values[0] : *global_best_fit;
    }
}


__global__ void calculate_fitness_update_bests(
    float *population,
    float *best_pos,
    float *best_fit,
    int n_pop,
    int dimension
) {
    int particle = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

     while(particle < n_pop){
        float fit = 0.0f;
        int start_index = particle * dimension;
        for (int d = 0; d < dimension; d++){
            float val = population[start_index + d];
            fit += (val-2.0f)*(val-2.0f);
        }
        if (fit < best_fit[particle]) {
            best_fit[particle] = fit;
            for (int d = 0; d < dimension; ++d) {
                int index = start_index + d;
                best_pos[index] = population[index];
            }
        }
        particle+=stride;
    }

}
__global__ void calc_fitness(
    float *population,
    float *fitness,
    int n_pop,
    int dimension
){
    int particle =  threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while(particle < n_pop){
        float fit = 0.0f;
        for (int d = 0; d < dimension; d++){
            float val = population[particle * dimension + d];
            fit += (val-2.0f)*(val-2.0f);
        }
        fitness[particle] = fit;
        particle+=stride;
    }
}

__global__ void update_best_positions(
    float *population,
    float *best_pos,
    float *fitness,
    float *best_fit,
    int n_pop,
    int dimension
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int particle;
    int size = n_pop * dimension;

    while (i < size){
        particle = i / dimension;
        best_pos[i] = fitness[particle] < best_fit[particle] ? population[i] : best_pos[i];
        i+=stride;
    }
}

__global__ void update_best_fitness(
    float *fitness,
    float *best_fit,
    int n_pop
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < n_pop){
        best_fit[i] = fitness[i] < best_fit[i] ? fitness[i] : best_fit[i];
        i+=stride;
    }
}

__global__ void init_population(float *population, float *best_pos, float *random_numbers, int seed, int popsize, int dimension) {

    int total_dimension = popsize * dimension;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < total_dimension) {
        float random_value = random_numbers[i];
        population[i] = (pso_upper_bound - pso_lower_bound)*random_value + pso_lower_bound;
        i += stride;
    }
}

__global__ void update_positions(
    float *population,
    float *velocity,
    float *best_pos,
    float *global_best,
    float *random_numbers,
    int n_pop,
    int dimension) {

    int total_dimension = n_pop * dimension;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int dim;

    while (i < total_dimension){

        float r1 = random_numbers[i];
        float r2 = random_numbers[i + total_dimension];

        dim = i % dimension;

        velocity[i] = (pso_inertial_weight*velocity[i]) + \
                        (pso_cognitive_param*r1*(best_pos[i] - population[i])) + \
                        (pso_social_param*r2*(global_best[dim] - population[i]));

        population[i] = population[i] + velocity[i];
        population[i] = fmaxf(fminf(population[i], pso_upper_bound), pso_lower_bound);

        i += stride;
    }
}

void print_population(float* dev_population, int pop_size) {
    float *h_population = (float*)malloc(pop_size * sizeof(float));
    cudaMemcpy(
        h_population,
        dev_population,
        pop_size * sizeof(float),
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

void print_best_local_fitness(float* dev_best_fit, int n_pop){
    float *h_best_fitness = (float*)malloc(n_pop * sizeof(float));
    cudaMemcpy(h_best_fitness, dev_best_fit, n_pop * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_pop; i++) {
        printf("best fitness[%d] = %f\n", i, h_best_fitness[i]);
    }
    free(h_best_fitness);
}

void print_best_position(float* dev_best_pos, int pop_size){
    float *h_best_pop = (float*)malloc(pop_size * sizeof(float));
    cudaMemcpy(
        h_best_pop,
        dev_best_pos,
        pop_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < pop_size; i++) {
        printf("best pop[%d] = %f\n", i, h_best_pop[i]);
    }
    free(h_best_pop);
}

void print_best_global_position(float* dev_global_best_pos, int dimension){
    float *h_global_best_pos = (float*)malloc(dimension * sizeof(float));
    cudaMemcpy(
        h_global_best_pos,
        dev_global_best_pos,
        dimension * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < dimension; i++) {
        printf("global best pos[%d] = %f\n", i, h_global_best_pos[i]);
    }
    free(h_global_best_pos);
}

int main(int argc, char **argv){

    float tempo_total_acumulado = 0.0f;
    float milliseconds = 0;
    int seed = SEED;
    float h_global_best_fitness;

    int dimension, n_pop, run_number, threads;
    bool verbose;

    if (argc > 1) {
        // If command line arguments are provided, use them to set parameters
        dimension = atoi(argv[1]);
        n_pop = atoi(argv[2]);
        threads = atoi(argv[3]);
        run_number = atoi(argv[4]);
        verbose = false;
    }
    else{
        dimension = DIMENSION;
        n_pop = NUMPARTICLES;
        run_number = 0;
        threads = NTHREADS;
        verbose = false;
    }

    const int pop_size = dimension * n_pop;
    int blocks = (pop_size + threads - 1) / threads;

    float *h_best_fitness = (float*)malloc(n_pop * sizeof(float));
    float *h_best_pop = (float*)malloc(pop_size * sizeof(float));

    int warp_size = 32;
    int reduction_threads = NTHREADS;
    if (n_pop < reduction_threads && n_pop > 0) {
        reduction_threads = ( (n_pop + warp_size - 1) / warp_size ) * warp_size;
        if (reduction_threads == 0) reduction_threads = warp_size;
    } else if (n_pop == 0){
        reduction_threads = warp_size;
    }

    unsigned int num_warps = (reduction_threads + warp_size - 1) / warp_size;
    size_t shared_mem_size = num_warps * (sizeof(float) + sizeof(int));


    if(verbose){
        printf("Running PSO with dimension=%d, n_pop=%d, threads=%d, blocks=%d, pop_size=%d\n", dimension, n_pop, threads, blocks, pop_size);
    }

    //memoria de constantes
    float h_lb = -5, h_ub = 5, h_IW = 0.7, h_CP = 0.5, h_SP = 0.5;
    cudaMemcpyToSymbol(pso_lower_bound, &h_lb, sizeof(float));
    cudaMemcpyToSymbol(pso_upper_bound, &h_ub, sizeof(float));
    cudaMemcpyToSymbol(pso_inertial_weight, &h_IW, sizeof(float));
    cudaMemcpyToSymbol(pso_cognitive_param, &h_CP, sizeof(float));
    cudaMemcpyToSymbol(pso_social_param, &h_SP, sizeof(float));

    // population, velocity
    float *dev_population, *dev_velocity, *dev_best_pos;
    cudaMalloc((void**)&dev_population, pop_size * sizeof(float));
    cudaMalloc((void**)&dev_velocity, pop_size * sizeof(float));
    cudaMalloc((void**)&dev_best_pos, pop_size * sizeof(float));

    float *dev_global_best_pos, *dev_global_best_fitness;
    cudaMalloc((void**)&dev_global_best_pos, dimension * sizeof(float));
    cudaMalloc((void**)&dev_global_best_fitness, sizeof(float));

    float *dev_fitness, *dev_best_fit;
    cudaMalloc((void**)&dev_fitness, n_pop * sizeof(float));
    cudaMalloc((void**)&dev_best_fit, n_pop * sizeof(float));

    float* dev_random_numbers;
    const int num_random_values = pop_size * 2;
    cudaMalloc((void**)&dev_random_numbers, num_random_values * sizeof(float));

    // inicializa velocidade e melhores posições com 0
    cudaMemset(dev_velocity, 0.0f, pop_size * sizeof(float));
    cudaMemset(dev_global_best_pos, 0.0f, dimension * sizeof(float));

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

    float h_inf = INF;
    cudaMemcpy(
        dev_global_best_fitness,
        &h_inf,
        sizeof(float),
        cudaMemcpyHostToDevice
    );

    curandGenerator_t rand_gen;
    curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rand_gen, seed);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    curandGenerateUniform(rand_gen, dev_random_numbers, num_random_values);

    init_population<<<blocks, threads>>>(
        dev_population,
        dev_best_pos,
        dev_random_numbers,
        seed,
        n_pop,
        dimension
    );

    calc_fitness<<<blocks, threads>>>(
        dev_population,
        dev_fitness,
        n_pop,
        dimension
    );

    update_best_positions<<<blocks, threads>>>(
        dev_population,
        dev_best_pos,
        dev_fitness,
        dev_best_fit,
        n_pop,
        dimension
    );

    int fbest_blocks = (n_pop + threads - 1) / threads;
    update_best_fitness<<<fbest_blocks, threads>>>(
        dev_fitness,
        dev_best_fit,
        n_pop
    );

    find_global_best_and_update_pos_warp<<<1, reduction_threads, shared_mem_size>>>(
        dev_best_fit,
        dev_best_pos,
        dev_global_best_fitness,
        dev_global_best_pos,
        n_pop,
        dimension
    );

    if (verbose){
        cudaMemcpy(&h_global_best_fitness, dev_global_best_fitness, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Global best fitness: %f\n", h_global_best_fitness);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    tempo_total_acumulado += milliseconds;

    int generations = 100;
    int iter = 0;
    cudaEventRecord(start);
    while (iter < generations) {

        curandGenerateUniform(rand_gen, dev_random_numbers, num_random_values);

        update_positions<<<blocks, threads>>>(
            dev_population,
            dev_velocity,
            dev_best_pos,
            dev_global_best_pos,
            dev_random_numbers,
            n_pop,
            dimension
        );

        cudaMemset(dev_fitness, 0.0f, n_pop * sizeof(float));
        calc_fitness<<<blocks, threads>>>(
            dev_population,
            dev_fitness,
            n_pop,
            dimension
        );

        update_best_positions<<<blocks, threads>>>(
            dev_population,
            dev_best_pos,
            dev_fitness,
            dev_best_fit,
            n_pop,
            dimension
        );

        update_best_fitness<<<fbest_blocks, threads>>>(
            dev_fitness,
            dev_best_fit,
            n_pop
        );

        if(verbose){
            print_fitness(dev_fitness, n_pop);
            print_best_local_fitness(dev_best_fit, n_pop);
            print_population(dev_population, pop_size);
        }

        find_global_best_and_update_pos_warp<<<1, reduction_threads, shared_mem_size>>>(
            dev_best_fit,
            dev_best_pos,
            dev_global_best_fitness,
            dev_global_best_pos,
            n_pop,
            dimension
        );

        if(verbose && (iter % 10 == 0)){
            cudaMemcpy(&h_global_best_fitness, dev_global_best_fitness, sizeof(float), cudaMemcpyDeviceToHost);
            printf("\nGlobal best fitness: %f\n", h_global_best_fitness);
            print_best_global_position(dev_global_best_pos, dimension);
        }

        iter++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    tempo_total_acumulado += milliseconds;

    cudaMemcpy(&h_global_best_fitness, dev_global_best_fitness, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%d,%d,%d,%d,%d,%f,%.5f\n", dimension, n_pop, threads, blocks, run_number, h_global_best_fitness,tempo_total_acumulado);

    cudaFree(dev_population);
    cudaFree(dev_velocity);
    cudaFree(dev_best_pos);
    cudaFree(dev_best_fit);
    cudaFree(dev_fitness);
    cudaFree(dev_global_best_pos);
    cudaFree(dev_global_best_fitness);

    free(h_best_fitness);
    free(h_best_pop);

    return 0;

}