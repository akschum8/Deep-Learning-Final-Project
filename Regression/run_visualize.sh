#!/bin/bash
#SBATCH --job-name=ultrasam_viz
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=viz_%j.out
#SBATCH --error=viz_%j.err

eval "$(conda shell.bash hook)"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate /home/cpsc4520_aks228/.conda/envs/UltraSam

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=.:$PYTHONPATH

cd /nfs/roberts/project/cpsc4520/cpsc4520_aks228/ultrasam_project/

python visualize_regression.py \
  --val_csv       data_train7z/train/csv_files/Reg-Three_2-2.700.csv \
  --frozen_ckpt   reg_runs/run1/best_model.pt \
  --lora_ckpt     reg_runs/lora/best_model.pt \
  --backbone_ckpt UltraSam/UltraSam.pth \
  --output_dir    visualizations \
  --num_images    8

echo "Exit code: $?"
