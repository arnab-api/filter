#!/bin/bash
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuH200x8      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bezl-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=finetuning
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

##SBATCH --mail-user=you@yourinstitution.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options


# initialize bash
source ~/.bashrc
# Initialize conda for shell script
eval "$(conda shell.bash hook)"

# activate conda environment
conda activate connection
conda env list

#export your required environment variables below
#################################################
export WANDB_API_KEY=$(cat wandb.key)

# cd to project directory
cd ~/Codes/Projects/retrieval
# run the script
python run_finetuning.py