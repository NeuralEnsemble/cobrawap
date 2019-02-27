import os

# determine working directory (data_challenge/ directory)
working_dir = os.path.dirname(os.path.realpath(__file__))

# path to experimental data
data_path = os.path.expanduser('~') + '/Sciebo/Data/WaveScalES/IDIBAPS/161101_rec07_Spontaneous_RH.smr'

# path to file with addition information; file must be called metadata.py
metadata_path = working_dir + '/metadata.py'

# path for generated data
output_path = working_dir + '/results/'

# path for generated figures
figure_path = working_dir + '/results/figures/'

# script path
script_path = working_dir + '/scripts/'
