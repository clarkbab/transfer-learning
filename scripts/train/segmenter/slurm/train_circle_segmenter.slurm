#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASETS=['CIRCLES']
MODEL_NAME="circle-segmenter"
n_EPOCHS=500
n_GPUS=1
n_NODES=1
n_WORKERS=4
n_SUBSET=None
RESUME=False
RESUME_CHECKPOINT=None
RUN_NAME="500epochs"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter_2d --slurm_job_id $SLURM_JOB_ID --model_name $MODEL_NAME --run_name $RUN_NAME --datasets $DATASETS --bottleneck $BOTTLENECK --n_epochs $n_EPOCHS \
    --n_gpus $n_GPUS --n_nodes $n_NODES --n_subset $n_SUBSET --n_workers $n_WORKERS --region $REGION --resume $RESUME --resume_checkpoint $RESUME_CHECKPOINT --use_logger $USE_LOGGER
