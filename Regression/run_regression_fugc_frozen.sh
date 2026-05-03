#!/bin/bash
#SBATCH --job-name=reg_fugc_frozen
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=reg_fugc_frozen_%j.out
#SBATCH --error=reg_fugc_frozen_%j.err

eval "$(conda shell.bash hook)"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate /home/cpsc4520_aks228/.conda/envs/UltraSam

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=.:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

cd /nfs/roberts/project/cpsc4520/cpsc4520_aks228/ultrasam_project/

python train_ultrasam_regression.py \
  --train_csv data_train7z/train/csv_files/Reg-Two_1-1.FUGC.csv \
  --val_csv   data_train7z/train/csv_files/Reg-Two_1-2.FUGC.csv \
  --checkpoint UltraSam/UltraSam.pth \
  --output_dir reg_runs/fugc_frozen \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_workers 2 \
  --freeze_backbone

echo "Exit code: $?"
