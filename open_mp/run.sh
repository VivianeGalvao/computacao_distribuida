# mpic++ main.cpp pso.cpp -o exec
# mpirun -np 6 ./exec

#!/bin/bash

# Define the number of runs
NUM_RUNS=10
# Define the number of processes for MPI
NUM_PROCS=12
# Define the output CSV file
OUTPUT_FILE="pso_execution_times_v3.csv"

# Check if the C++ source file exists
if [ ! -f "pso.cpp" ]; then
    echo "Error: pso.cpp not found!"
    exit 1
fi

# Compile the C++ program
echo "Compiling pso.cpp..."
if ! g++ -fopenmp -o pso main.cpp pso.cpp; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful. Executable 'pso' created."

# Add a header to the CSV file
echo "dimension,n_pop,n_thread,run_number,execution_time" > "$OUTPUT_FILE"

echo "Starting $NUM_RUNS runs..."

# Use a POSIX-compliant for loop
for dimension in 10 100 1000 10000
do
    for n_pop in 10 50 100 500
    do
        for threads in $(seq 1 $NUM_PROCS)        
        do
            for i in $(seq 1 $NUM_RUNS)
            do
                echo "Running PSO with dimension $dimension and population size $n_pop and threads $threads..."
                echo "Running iteration $i of $NUM_RUNS..."

                # g++ -fopenmp -o pso main.cpp pso.cpp
                export OMP_NUM_THREADS=$threads
                start_time=$(date +%s%3N)
                # Capture the output of the time command by redirecting stderr to stdout
                #TIME_STR=$( { time -q -f '%e' ./pso $dimension $n_pop >/dev/null; } 2>&1 )
                ./pso $dimension $n_pop $threads >/dev/null
                # Trim leading/trailing whitespace from the captured time string
                #TIME_STR=$(echo "$TIME_STR" | xargs)
                end_time=$(date +%s%3N)
                
                TIME_STR=$((end_time - start_time))

                echo "Execution time: $TIME_STR milliseconds"
                
                # Add the run number and execution time to the CSV
                echo "$dimension,$n_pop,$threads,$i,$TIME_STR" >> "$OUTPUT_FILE"
            done
        done
    done
done

echo "All runs completed. Results stored in $OUTPUT_FILE."