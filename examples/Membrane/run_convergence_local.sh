#!/bin/bash

# Convergence study script for Membrane simulation
# Launches multiple Julia processes on local computer with varying number of divisions
# All simulations run simultaneously in the background and are independent from each other

# Define the array of division numbers to test
divisions=(10 20 30 40 50 60)

# Get the directory where this script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get number of available cores (optional: limit with MAX_PARALLEL_JOBS)
num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
max_parallel=${MAX_PARALLEL_JOBS:-$num_cores}

echo "========================================"
echo "Starting Membrane Convergence Study"
echo "Launching multiple Julia processes locally"
echo "========================================"
echo "Time: $(date)"
echo "Available cores: $num_cores"
echo "Max parallel jobs: $max_parallel"
echo ""

# Array to store process IDs
declare -a pids

# Counter for parallel job management
job_count=0

# Launch Julia process for each division count
for ndiv in "${divisions[@]}"
do
  # Wait if we've reached max parallel jobs
  while [ $(jobs -r | wc -l) -ge $max_parallel ]; do
    sleep 1
  done
  
  echo "Launching Julia process for ndivisions=$ndiv"
  
  # Launch the Julia simulation in the background
  (
    echo "========================================"
    echo "Starting Membrane Convergence Simulation"
    echo "ndivisions = $ndiv"
    echo "Process ID: $$"
    echo "Time: $(date)"
    echo "========================================"
    
    # Set environment variables
    export JULIA_PKG_OFFLINE=true
    
    # Execute the Julia script
    cd "$script_dir"
    julia --project=.. -e "
      # Load the problem parameters and modify ndivisions
      include(\"Membrane.jl\")
      problem_data = merge(problem_data, (ndivisions = $ndiv,))
      
      # Solve the problem
      solve_problem(problem_data)
    " > "Membrane_Convergence_ndiv_${ndiv}.out" 2> "Membrane_Convergence_ndiv_${ndiv}.err"
    
    exit_code=$?
    
    echo ""
    echo "========================================"
    if [ $exit_code -eq 0 ]; then
      echo "Simulation for ndivisions=$ndiv completed successfully"
    else
      echo "Simulation for ndivisions=$ndiv failed with exit code: $exit_code"
    fi
    echo "Time: $(date)"
    echo "========================================"
    
    exit $exit_code
  ) &
  
  pids+=($!)
  job_count=$((job_count + 1))
done

echo ""
echo "========================================"
echo "All Julia processes launched!"
echo "========================================"
echo "Total processes: $job_count"
echo "Process IDs: ${pids[@]}"
echo ""

# Wait for all background processes to complete
echo "Waiting for all processes to complete..."
failed_count=0
for i in "${!pids[@]}"; do
  pid=${pids[$i]}
  ndiv=${divisions[$i]}
  
  if wait $pid; then
    echo "✓ Process $pid (ndivisions=$ndiv) completed successfully"
  else
    echo "✗ Process $pid (ndivisions=$ndiv) failed"
    failed_count=$((failed_count + 1))
  fi
done

echo ""
echo "========================================"
echo "Convergence study completed!"
echo "========================================"
echo "Time: $(date)"

if [ $failed_count -eq 0 ]; then
  echo "All simulations completed successfully"
else
  echo "WARNING: $failed_count simulation(s) failed"
fi

echo ""
echo "To view results:"
echo "  ls -la Membrane_Convergence_ndiv_*.out"
echo "  ls -la Membrane_Convergence_ndiv_*.err"
echo ""
echo "To view simulation output:"
echo "  cat Membrane_Convergence_ndiv_10.out"
echo "  cat Membrane_Convergence_ndiv_10.err"
