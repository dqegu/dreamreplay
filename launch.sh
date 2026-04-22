#!/bin/bash -l
#SBATCH -p l4,scavenger_l4,ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@soton.ac.uk
#SBATCH -o slurm-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spens-seq

export OMP_NUM_THREADS=8
export IRP_BASE_DIR=/iridisfs/home/ao1g22/comp6228/irp

export LD_LIBRARY_PATH=/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cublas/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cufft/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/curand/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cusolver/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/cusparse/lib:/home/ao1g22/.conda/envs/spens-seq/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

cd /iridisfs/home/ao1g22/comp6228/irp

python run_all.py --seeds 0 7 42 --n_train 4000 --n_test 1000
