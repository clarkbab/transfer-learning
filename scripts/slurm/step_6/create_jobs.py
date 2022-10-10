import subprocess

regions = '0-16'
models = ['localiser', 'segmenter']
model_abbrs = ['LOC', 'SEG']

for model, model_abbr in zip(models, model_abbrs):
    script = f'scripts/slurm/step_6/{model}_template.slurm'
    datasets = [f'HN1-{model_abbr}', f'HNPCT-{model_abbr}', f'HNSCC-{model_abbr}', f'OPC-{model_abbr}']

    # Create slurm command.
    export = f'ALL,DATASETS={datasets}'
    command = f'sbatch --array={regions} --export={export} {script}' 
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
