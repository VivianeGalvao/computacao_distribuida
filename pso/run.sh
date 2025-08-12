#g++ main.cpp pso.cpp -o exec
#./exec

#!/bin/bash

# Define the number of runs
NUM_RUNS=30
# Define the number of processes for MPI
NUM_PROCS=4
# Define the output CSV file
OUTPUT_FILE="pso_execution_times_10000.csv"

# Check if the C++ source file exists
if [ ! -f "pso.cpp" ]; then
    echo "Error: pso.cpp not found!"
    exit 1
fi

# Compile the C++ program
echo "Compiling pso.cpp..."
if ! g++ -o pso main.cpp pso.cpp; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful. Executable 'pso' created."

# Add a header to the CSV file
echo "run_number,execution_time" > "$OUTPUT_FILE"

echo "Starting $NUM_RUNS runs..."

# Loop to run the program multiple times
for i in $(seq 1 $NUM_RUNS)
do
    echo "Running iteration $i of $NUM_RUNS..."
    
    # Capture the output of the time command by redirecting stderr to stdout
    TIME_STR=$( { time -f '%e' ./pso >/dev/null; } 2>&1 )
    
    # Trim leading/trailing whitespace from the captured time string
    TIME_STR=$(echo "$TIME_STR" | xargs)
    
    echo "Execution time: $TIME_STR seconds"
    
    # Add the run number and execution time to the CSV
    echo "$i,$TIME_STR" >> "$OUTPUT_FILE"
done

echo "All runs completed. Results stored in $OUTPUT_FILE."