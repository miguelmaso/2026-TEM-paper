#!/bin/bash
#SBATCH --job-name=Wrinkles
#SBATCH --output=%x-output-job_%j.out
#SBATCH --error=%x-output-job_%j.err
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=1

##Optional - Required memory in MB per core. Defaults are 1GB per core.
#SBATCH --mem-per-cpu=8192

##Optional - Estimated execution time
##Acceptable time formats include  "minutes",   "minutes:seconds",
##"hours:minutes:seconds",   "days-hours",   "days-hours:minutes" ,"days-hours:minutes:seconds".
##SBATCH --time=5-0

########### Further details -> man sbatch ##########

julia --project=.. --threads 8 ./Wrinkles.jl
