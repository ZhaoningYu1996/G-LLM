#!/bin/bash

#SBATCH --nodes=1   # Number of nodes to use
#SBATCH --ntasks-per-node=4   # Use 8 processor cores per node 
#SBATCH --time=3-8:0:0   # Walltime limit (DD-HH:MM:SS)
#SBATCH --mem=64G   # Maximum memory per node
#SBATCH --gres=gpu:a100:1   # Required GPU hardware
#SBATCH --job-name="llm-exp"   # Job name to display in squeue
#SBATCH --mail-user=znyu@iastate.edu   # Email address
#SBATCH --mail-type=BEGIN   # Send an email when the job starts
#SBATCH --mail-type=END   # Send an email when the job ends
#SBATCH --mail-type=FAIL   # Send an email if the job fails

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load python/3.10.10-zwlkg4l

source /work/LAS/hygao-lab/znyu/G-LLM/llm/bin/activate

python unsloth_sft.py