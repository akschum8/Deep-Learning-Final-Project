#!/bin/bash
#SBATCH --job-name=det_fibroid_frozen
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=det_fibroid_frozen_%j.out
#SBATCH --error=det_fibroid_frozen_%j.err

eval "$(conda shell.bash hook)"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate /home/cpsc4520_aks228/.conda/envs/UltraSam

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=.:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

cd /nfs/roberts/project/cpsc4520/cpsc4520_aks228/ultrasam_project/

python split_detection_csv.py \
  --csv       data_train7z/train/csv_files/Det-3.uterine_fibroid_det.csv \
  --train_out data_train7z/train/csv_files/Det-3.uterine_fibroid_train.csv \
  --val_out   data_train7z/train/csv_files/Det-3.uterine_fibroid_val.csv \
  --val_frac 0.2 --seed 42

python train_ultrasam_detection.py \
  --train_csv data_train7z/train/csv_files/Det-3.uterine_fibroid_train.csv \
  --val_csv   data_train7z/train/csv_files/Det-3.uterine_fibroid_val.csv \
  --checkpoint UltraSam/UltraSam.pth \
  --output_dir det_runs/fibroid_frozen \
  --epochs 20 \
  --batch_size 2 \
  --num_workers 2 \
  --freeze_backbone

echo "Exit code: $?"
