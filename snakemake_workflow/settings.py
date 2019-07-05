import os

# determine working directory (data_challenge/ directory)
working_dir = os.path.dirname(os.path.realpath(__file__))

# path to experimental data
data_path = os.path.expanduser('~') + '/Sciebo/own/Data/WaveScalES/'

# path to file with addition information; file must be called metadata.py
metadata_path = working_dir + '/metadata.py'

# path for generated data
output_path = os.path.join(os.path.expanduser('~'), 'ProjectsData/wavescalephant/')

# path for generated figures
figure_path = os.path.join(output_path, 'figures/')

# script path
script_path = working_dir + '/scripts/'
