#!/bin/bash
#PBS -N leopard
#PBS -P wo64
#PBS -r y
#PBS -q gpuvolta
#PBS -l walltime=01:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -M ez216@exeter.ac.uk
#PBS -m e


###########################
#load modules for gpu support
module load cuda
module load cudnn
module load nccl
module load openmpi

# setup conda environment
# -- change the path to your own conda directory
source /scratch/wo64/ez3336/conda.sh
conda init bash
conda activate leopard

# run the bechmark over one GPUs
source /scratch/wo64/ez3336/2022-APAC-HPC-AI/Deep_Learning_Based_DNA_Sequence_Fast_Decoding/train.sh

