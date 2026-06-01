#!/bin/bash
#SBATCH --job-name=Membrane_Convergence
#SBATCH --output=%x-ndiv_%a-output-job_%j.out
#SBATCH --error=%x-ndiv_%a-output-job_%j.err
#SBATCH --ntasks=1                     # Serial execution (single task)
#SBATCH --cpus-per-task=1              # 1 CPU per task
#SBATCH --partition=R630               # Partition (verify with sinfo/sview)
#SBATCH --time=1-00:00:00              # Time limit (days-hours:minutes:seconds)
#SBATCH --mem-per-cpu=36864            # Memory per CPU (MB)

# --- Load Modules (if needed) ---
# module load julia/1.10.2

# --- Configure Julia (only needed first time, then comment out) ---
# julia --project=.. config_mpi.jl
export JULIA_PKG_OFFLINE=true

# --- Get ndivisions from command line argument ---
ndivisions=${1:-10}

echo "========================================"
echo "Starting Membrane Convergence Simulation"
echo "ndivisions = $ndivisions"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "========================================"

# --- Execute the Julia script ---
srun julia --project=.. -e "
  # Load the problem parameters and modify ndivisions
  include(\"Membrane.jl\")
  problem_data = merge(problem_data, (ndivisions = $ndivisions,))
  
  # Solve the problem
  solve_problem(problem_data)
"

exit_code=$?

echo ""
echo "========================================"
if [ $exit_code -eq 0 ]; then
  echo "Simulation completed successfully"
else
  echo "Simulation failed with exit code: $exit_code"
fi
echo "Time: $(date)"
echo "========================================"

exit $exit_code
