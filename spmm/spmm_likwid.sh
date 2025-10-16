#!/bin/bash
#SBATCH --job-name=spmm_likwid
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu
#SBATCH --output=perf_%j.out
#SBATCH --error=perf_%j.err
#SBATCH --mem=100G

# 1. module load intel/2024.1.0
# 2. intall likwid
# 3. import LIKWID_ROOT
# 4. export num threads

module load intel/2024.1.0
module load spack/0.22

spack env activate gcc
spack env st
spack load likwid

export LIKWID_ROOT=/global/common/software/spackecp/perlmutter/v0.22/96029/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/likwid-5.3.0-4jj3tikmshwg6nubpyirxkqhids5fu4t

cd $SCRATCH/SpMM_benchmark/spmm

# List of matrices and number of columns
declare -A matrices
matrices=(
  ["data/5Mx5M_50M.mtx"]=16
  ["data/333SP.mtx"]=16
  ["data/asia_osm.mtx"]=16
  ["data/com-LiveJournal.mtx"]=16
  ["data/hugebubbles-00010.mtx"]=16
  ["data/rajat31.mtx"]=16
  ["data/road_usa.mtx"]=16
  ["data/Spielman_k300.mtx"]=16
)

# Output CSV
summary_csv="spmm_likwid_summary.csv"

# Header
# echo "Matrix,CSR DP [MFLOP/s],CSR RETIRED_SSE_AVX_FLOPS_ALL,CSR Runtime [s],MKL DP [MFLOP/s],MKL RETIRED_SSE_AVX_FLOPS_ALL,MKL Runtime [s],CSB DP [MFLOP/s],CSB RETIRED_SSE_AVX_FLOPS_ALL,CSB Runtime [s]" > $summary_csv
echo "Matrix,CSR DP [MFLOP/s],MKL DP [MFLOP/s],CSB DP [MFLOP/s],CSR Runtime [s],MKL Runtime [s],CSB Runtime[s],CSR Count,MKL Count,CSB Count" > $summary_csv
# Set number of OpenMP threads
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
cores="0-7"

# Loop over matrices
for mat in "${!matrices[@]}"; do
    cols=${matrices[$mat]}
    name=$(basename "$mat" .mtx)
    echo ">>> Running LIKWID on $name ..."

    # Run FLOPS_DP measurement
    likwid_out=$(likwid-perfctr -m -C $cores -g FLOPS_DP ./spmm "$mat" $cols 2>/dev/null)

    # Extract FLOPs from RETIRED_SSE_AVX_FLOPS_ALL row
    ALL_flops=($(echo "$likwid_out" | awk '/RETIRED_SSE_AVX_FLOPS_ALL/ {sum=0; for(i=3;i<=NF;i++) sum+=$i; print sum}'))
    CSR_count=${ALL_flops[0]}
    MKL_count=${ALL_flops[1]}
    CSB_count=${ALL_flops[2]}

    
    runtimes=($(echo "$likwid_out" | awk '/RDTSC/ {for(i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/) print $i}'))
    CSR_RDTSC=${runtimes[0]}
    MKL_RDTSC=${runtimes[2]}
    CSB_RDTSC=${runtimes[4]}
    
    DP_flops=($(echo "$likwid_out" | awk '/DP/ {sum=0; for(i=3;i<=NF;i++) sum+=$i; print sum}'))
    CSR_DP=${DP_flops[1]}
    MKL_DP=${DP_flops[3]}
    CSB_DP=${DP_flops[5]}
    
    # Append to CSV
    echo "$name,$CSR_DP,$MKL_DP,$CSB_DP,$CSR_RDTSC,$MKL_RDTSC,$CSB_RDTSC,$CSR_count,$MKL_count,$CSB_count" >> $summary_csv

done

echo "=== Done. Summary written to $summary_csv ==="
