# nvcc pso.cu -o pso -arch=sm_60 -DFLOAT -DFORUM
# time ./pso

# Define the number of runs
NUM_RUNS=10
# Define the output CSV file
OUTPUT_FILE="pso_execution_times_cuda_p2.csv"

# Check if the C++ source file exists
if [ ! -f "pso.cu" ]; then
    echo "Error: pso.cu not found!"
    exit 1
fi

# Compile the C++ program
echo "Compiling pso.cu..."
if ! nvcc pso.cu -o pso -arch=sm_60 -DFLOAT -DFORUM; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful. Executable 'pso' created."

# Add a header to the CSV file
echo "dimension,n_pop,n_thread,n_block,run_number,best_value,execution_time" > "$OUTPUT_FILE"

echo "Starting $NUM_RUNS runs..."

# Use a POSIX-compliant for loop
for dimension in 512 1024 2048 4096 8192
do
    for n_pop in 64 128 256 512
    do
        for threads in 128 256 512
        do
            nvcc pso.cu -o pso
            for i in $(seq 1 $NUM_RUNS)
            do
                echo "Running: Dim=$dimension, Pop=$n_pop, Threads=$threads, Run=$i/$NUM_RUNS"

                # Executa o programa e anexa a saída formatada ao arquivo CSV
                # Assumindo que seu programa foi modificado para imprimir no formato CSV
                ./pso $dimension $n_pop $threads $i >> "$OUTPUT_FILE"
            done
        done
    done
done

echo "Experimentos finalizados. Resultados em $OUTPUT_FILE"

echo "All runs completed. Results stored in $OUTPUT_FILE."