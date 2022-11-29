#!/bin/bash
#PBS -N leopard_2gpu
#PBS -P wo64
#PBS -r y
#PBS -q gpuvolta
#PBS -l walltime=01:30:00
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=100GB
#PBS -M ez216@exeter.ac.uk
#PBS -m e


###########################
#load modules for gpu support
module load cuda
module load cudnn
module load nccl
module load openmpi
#module load horovod/0.22.1

# setup conda environment
# -- change the path to your own conda directory
source /scratch/wo64/ez3336/conda.sh
conda init bash
conda activate leopard

# run the bechmark over 2 GPUs
# -- change the path to your own
source /scratch/wo64/ez3336/2022-APAC-HPC-AI/Deep_Learning_Based_DNA_Sequence_Fast_Decoding/multi_gpus_train.sh


