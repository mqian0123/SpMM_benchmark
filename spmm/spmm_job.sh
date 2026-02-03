#!/bin/bash
#SBATCH -A m4012
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:45:00
#SBATCH -J spmm_k
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --cpu-freq=performance


set -euo pipefail

mkdir -p logs results

module load PrgEnv-gnu

# =========================
# SET K HERE (manually)
# =========================
K=16   # <-- change to 1,2,4,8,16,32,64
# =========================

MATDIR="$(pwd)"


MATRICES=(
  "$MATDIR/data/333SP.mtx"
  # "road_usa.mtx"
  # "rajat31.mtx"
)

THREADS=(1 2 4 8 16 32 64)

export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== Job ${SLURM_JOB_ID}: rebuilding for RHSDIM=${K} ==="

# Use a per-job build dir to avoid collisions if you submit multiple jobs at once
BUILD_DIR="build_RHSDIM${K}_job${SLURM_JOB_ID}"
BIN="spmm_RHSDIM${K}"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy build inputs (add more files here if your project has them)
cp -f spmm.cpp Makefile.gnu "$BUILD_DIR"/
cp -r csb_library "$BUILD_DIR"/


pushd "$BUILD_DIR" >/dev/null
make -f Makefile.gnu clean || true
make -f Makefile.gnu TARGET="$BIN"
popd >/dev/null

OUTCSV="results/spmm_RHSDIM${K}.csv"
if [[ ! -f "$OUTCSV" ]]; then
  echo "timestamp,job_id,k,matrix,threads,csr_s,mkl_s,csb_s,exit_code,runlog" > "$OUTCSV"
fi

for mat in "${MATRICES[@]}"; do
  if [[ ! -f "$mat" ]]; then
    echo "WARNING: matrix file not found: $mat (skipping)"
    continue
  fi

  for t in "${THREADS[@]}"; do
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$t" -gt "$SLURM_CPUS_PER_TASK" ]]; then
      continue
    fi

    base="$(basename "${mat%.mtx}")"
    RUNLOG="logs/run_k${K}_${base}_t${t}_job${SLURM_JOB_ID}.log"

    echo
    echo "----- RHSDIM=${K} matrix=${mat} threads=${t} -----"

    set +e
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"

    srun -N 1 -n 1 \
      numactl --interleave=all \
      "./${BUILD_DIR}/${BIN}" "$mat" "$K" "$t" >"$RUNLOG" 2>&1
    rc=$?
    set -e



    csr_s="$(awk -F'[: ]+' '/spmm_csr time:/ {gsub(/s/,"",$NF); print $NF; exit}' "$RUNLOG")"
    mkl_s="$(awk -F'[: ]+' '/spmm_mkl time:/ {gsub(/s/,"",$NF); print $NF; exit}' "$RUNLOG")"
    csb_s="$(awk -F'[: ]+' '/spmm_csb time:/ {gsub(/s/,"",$NF); print $NF; exit}' "$RUNLOG")"

    echo "$(date -Is),${SLURM_JOB_ID},${K},${mat},${t},${csr_s:-},${mkl_s:-},${csb_s:-},${rc},${RUNLOG}" >> "$OUTCSV"
    echo "csr=${csr_s:-NA}s  mkl=${mkl_s:-NA}s  csb=${csb_s:-NA}s  (rc=${rc})"
  done
done

echo
echo "Done. CSV: $OUTCSV"
