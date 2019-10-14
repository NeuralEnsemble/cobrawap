import os

# determine working directory (data_challenge/ directory)
working_dir = os.path.dirname(os.path.realpath(__file__))

# path to experimental data
data_path = os.path.expanduser('~') + '/Sciebo/own/Data/WaveScalES/'

# path for generated data
output_path = os.path.join(os.path.expanduser('~'), 'ProjectsData/wave_analysis_pipeline/')
