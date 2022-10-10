#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 5-0:00:00

version=$(python --version)
echo $version

# 'DATASETS', 'N_EPOCHS', 'N_TRAIN' and 'TEST_FOLD' passed by 'create_jobs' script.
REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Brain'             # 2
    'BrainStem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Mandible'          # 8
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
    'OralCavity'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Submandibular_L'   # 15
    'Submandibular_R'   # 16
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-${REGION}"
N_GPUS=1
N_NODES=1
N_WORKERS=4
RESUME=False
RESUME_CHECKPOINT=None
USE_LOGGER=False        # Set 'True' to use 'wandb' logging. Must set this up first (https://wandb.ai).
RUN_NAME="public-1gpu-150epochs"

# Check if 'N_TRAIN' training samples exist for this region.
N_TRAIN_MAX = $(python ../get_n_train_max.py \
    --datasets $DATASETS \
    --region $REGION \
    --test_fold $TEST_FOLD)

if [ \( "$N_TRAIN" == "all" \) -o \( "$N_TRAIN" -ge "$N_TRAIN_MAX" \) ]; then
    echo "N_TRAIN ($N_TRAIN) >= N_TRAIN_MAX ($N_TRAIN_MAX). Skipping."
    exit 0
fi

python ../train_segmenter.py \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --datasets $DATASETS \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --region $REGION \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
