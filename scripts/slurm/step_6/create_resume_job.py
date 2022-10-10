import subprocess

# Select 'region', 'model' and 'model_abbr'.
region = '2'
model = 'localiser'
# model = 'segmenter'
model_abbr = 'LOC'
# model_abbr = 'SEG'
script = f'scripts/slurm/step_6/resume_{model}_template.slurm'
datasets = [f'HN1-{model_abbr}', f'HNPCT-{model_abbr}', f'HNSCC-{model_abbr}', f'OPC-{model_abbr}']

# Create slurm command.
export = f'ALL,DATASETS={datasets}'
command = f'sbatch --array={region} --export={export} {script}' 
print(command)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
process.communicate()
