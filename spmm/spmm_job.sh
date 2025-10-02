#!/bin/bash
#SBATCH --job-name=spmm_perf
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu
#SBATCH --licenses=scratch
#SBATCH --output=perf_%j.out
#SBATCH --error=perf_%j.err

module load PrgEnv-intel
export PATH=/global/common/software/nersc9/intel/oneapi/advisor/2024.1/bin64:$PATH # Adds intel advisor

# Specifies number of threads tested
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Need to specify number of columns in B
MATRICES=(
    "road_usa.mtx 8"
    # "asia_osm.mtx 8"
    # "333SP.mtx 8"
)

for MAT in "${MATRICES[@]}"; do
    NAME=$(echo $MAT | awk '{print $1}' | sed 's/.mtx//')
    echo ">>> Running Intel Advisor tripcounts for $NAME"

    srun -n 1 -c ${OMP_NUM_THREADS} advixe-cl --collect=tripcounts \
        --flop --stacks --cache-simulation \
        --project-dir=advisor_${NAME} \
        -- ./spmm $MAT
done

