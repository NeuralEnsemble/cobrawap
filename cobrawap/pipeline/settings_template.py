import os

# path for generated data
output_path = os.path.join(os.path.expanduser('~'), 'path/to/pipeline/output/folder/')

# optional alternative path for config files
# directory must contain stageXY_<stage-name>/config_<PROFILE>.yaml
# if None uses the pipeline working directory
configs_dir = None
