#!/bin/bash

# Convergence study script for Membrane simulation
# Submits multiple Membrane.jl simulations with varying number of divisions via sbatch
# All simulations run simultaneously and are independent from each other

# Define the array of division numbers to test
divisions=(10 20 30 40 50 60)

# Get the directory where this script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Starting Membrane Convergence Study"
echo "Submitting jobs via sbatch"
echo "========================================"
echo "Time: $(date)"

# Array to store job IDs
declare -a job_ids

# Submit sbatch job for each division count
for ndiv in "${divisions[@]}"
do
  echo "Submitting job for ndivisions=$ndiv"
  
  # Submit the sbatch job and capture the job ID
  job_id=$(sbatch "${script_dir}/run_convergence_single.sh" "$ndiv" | awk '{print $NF}')
  job_ids+=("$job_id")
  
  echo "  Job ID: $job_id"
done

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo "Job IDs: ${job_ids[@]}"
echo ""
echo "To check job status:"
echo "  squeue -j ${job_ids[0]}"
echo "  squeue -l"
echo ""
echo "To view results:"
echo "  ls -la Membrane_Convergence_ndiv_*.out"
echo "  ls -la Membrane_Convergence_ndiv_*.err"
