#!/bin/bash -l
#SBATCH -p l4,scavenger_l4,ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@soton.ac.uk
#SBATCH -o slurm-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spens-seq

# Optional: avoid TF grabbing all CPU threads
export OMP_NUM_THREADS=8

python run_seq_replay.py
