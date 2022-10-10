#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-01:00:00

version=$(python --version)
echo $version

join () {
  local IFS="$1"
  shift
  echo "$*"
}

# 'DATASET' passed by 'create_jobs' script.
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
ANONYMISE=False     # Use ANONYMISE=True if moving data offsite for training.
SIZE=None
SPACING="(1,1,2)"
REGIONS=$(join , "${REGIONS[@]}")

python ../convert_to_training \
    --dataset $DATASET \
    --dest_dataset $DATASET \
    --regions $REGIONS \
    --size $SIZE \
    --spacing $SPACING
