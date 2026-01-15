#!/bin/bash
#SBATCH --job-name=PatchTest_Julia
#SBATCH --output=%x-output-job_%j.out
#SBATCH --error=%x-output-job_%j.err
#SBATCH --ntasks=8                    # <--- OBLIGATORIO: 2x2x2 = 8
#SBATCH --partition=R630              # Partición a usar (sview, Visible tabs -> Nodes)
#SBATCH --cpus-per-task=1             # 1 CPU por proceso MPI
#SBATCH --time=1-00:00:00             # Límite de tiempo (dias-horas:minutos:segundos)
#SBATCH --mem-per-cpu=4096            # Memoria por hilo (MB)

# --- Cargar Módulos ---
# module load julia/1.10.2
# Si GridapPETSc va a usar el MPI del sistema, carga también:
# module load openmpi petsc  

# --- Configurar Julia (Solo necesario la primera vez) ---
# Descomenta la siguiente línea la PRIMERA vez que lances esto, luego coméntala.
julia --project=.. config_mpi.jl
export JULIA_PKG_OFFLINE=true

# --- Ejecución ---
# 'srun' es el lanzador nativo de Slurm, no es necesario usar 'mpiexec' o 'mpirun'.
srun julia --project=.. ./PatchTest-distributed.jl
