#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/adjoint_samplers
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=2
#SBATCH --job-name=adjs
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/project/aip-aspuru/aburger/adjoint_samplers/outslurm/slurm-%j.txt 
#SBATCH --error=/project/aip-aspuru/aburger/adjoint_samplers/outslurm/slurm-%j.txt

# get environment variables
source .env

# activate venv
source .venv/bin/activate

#module load cuda/12.6
#module load gcc/12.3

# append command to slurmlog.txt
echo "sbatch scripts/killarney.sh $@ # $SLURM_JOB_ID" >> slurmlog.txt

echo `date`: Job $SLURM_JOB_ID is allocated resources.

srun .venv/bin/python "$@"
