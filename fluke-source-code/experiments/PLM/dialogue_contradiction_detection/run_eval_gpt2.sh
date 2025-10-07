#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=64G

#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --constraint="dlg4|dlg5"
#SBATCH -A punim1194

module load foss/2022a CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10
# activate my conda env
eval "$(conda shell.bash hook)"
conda activate lora

export PYTHONPATH="${PYTHONPATH}:/data/gpfs/projects/punim1194/ood-phase2/pretrained_language_models/dialogue_contradiction_detection"
export WANDB_PROJECT="ood-plms"
export WANDB_RUN_NAME="gpt2_dialog-contra-detect_${SLURM_JOB_ID}"

python eval_gpt2.py
