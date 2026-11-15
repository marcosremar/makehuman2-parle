#!/bin/bash
#SBATCH --job-name=cabeca-teste
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --tmp=30G
#SBATCH --output=/home/maramosp/cabeca-teste/log-%j.out
# Quick test of a stylized head: SDXL-Turbo image + Hunyuan3D-2mini shape, inside the PyTorch container. Scratch removed at the end.
set -uo pipefail
WORKDIR=/storage/scratch/$LOGNAME/$SLURM_JOB_ID
mkdir -m 700 -p $WORKDIR
trap 'rm -rf $WORKDIR' EXIT
export WORKDIR TMPDIR=$WORKDIR HF_HOME=/home/maramosp/hf_cache PIP_CACHE_DIR=$WORKDIR/pip APPTAINER_CACHEDIR=$WORKDIR/ac APPTAINER_TMPDIR=$WORKDIR
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 30 -i $SLURM_JOB_GPUS > /home/maramosp/cabeca-teste/gpu-$SLURM_JOB_ID.csv &
SIF=/home/maramosp/containers/pytorch-2.4.1-cuda12.1.sif
mkdir -p /home/maramosp/containers
[ -f $SIF ] || apptainer pull $SIF docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
apptainer exec --nv --bind /storage/scratch,/home/maramosp $SIF bash /home/maramosp/cabeca-teste/dentro.sh
echo FIM
