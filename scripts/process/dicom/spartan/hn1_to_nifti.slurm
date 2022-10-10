#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-2:00:00
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

bash $SCRIPT_DIR/process/dicom/manual/hn1_to_nifti

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")"; cd ..; pwd -P )
cd $parent_path

DATASET="HN1"
REGIONS="('BrachialPlexus_L','BrachialPlexus_R','Brain','BrainStem','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Mandible','OpticNerve_L','OpticNerve_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R')"
ANONYMISE=False

python convert_to_nifti --dataset $DATASET --regions $REGIONS --anonymise $ANONYMISE