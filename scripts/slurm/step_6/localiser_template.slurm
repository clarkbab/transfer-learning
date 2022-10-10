#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 5-0:00:00

version=$(python --version)
echo $version

# 'DATASETS' passed by 'create_jobs' script.
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
MODEL_NAME="localiser-${REGION}"
N_EPOCHS=150
N_GPUS=1
N_NODES=1
N_WORKERS=4
RESUME=False
RESUME_CHECKPOINT=None
USE_LOGGER=False        # Set 'True' to use 'wandb' logging. Must set this up first (https://wandb.ai).
RUN_NAME="public-1gpu-150epochs"

python ../train_localiser \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --datasets $DATASETS \
    --n_epochs $n_EPOCHS \
    --n_gpus $n_GPUS \
    --n_nodes $n_NODES \
    --n_workers $n_WORKERS \
    --region $REGION \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --use_logger $USE_LOGGER
