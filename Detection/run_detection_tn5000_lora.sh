#!/bin/bash
#SBATCH --job-name=det_tn5000_lora
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=det_tn5000_lora_%j.out
#SBATCH --error=det_tn5000_lora_%j.err

eval "$(conda shell.bash hook)"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate /home/cpsc4520_aks228/.conda/envs/UltraSam

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=.:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

cd /nfs/roberts/project/cpsc4520/cpsc4520_aks228/ultrasam_project/

python train_ultrasam_detection_lora.py \
  --train_csv data_train7z/train/csv_files/Det-1.TN5000_train.csv \
  --val_csv   data_train7z/train/csv_files/Det-1.TN5000_val.csv \
  --checkpoint UltraSam/UltraSam.pth \
  --output_dir det_runs/tn5000_lora \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --num_workers 2

echo "Exit code: $?"
