# Stage 1 - Data Curation
<!-- ToDo: How to mark the capabilities of the dataset -->

The data curation stage is very specific to the given dataset. Therefore,
each dataset essentially requires its own block, which should handle the
following actions:
* loading the data
* transforming the data into the Neo format
* adding the metadata from the config file
* transform AnalogSignal to ImageSequence or vice versa
* (run custom checks on the data, and annotate results (e.g. bad electrodes))
* storing the neo object in the Nix format

## Namespace
For a dataset with the unique identifying name <data_name>,
use the following naming scheme
Rule: <data_name>
Script: curate_<data_name>
Configfile: config_<data_name>

## Neo structure
The data should be contained in one AnalogSignal object with dimensions
(time, channel). This AnalogSignal needs to be located in the
first Segment. When the data is optical data represented as ImageSequence, it
should also be transformed into a corresponding AnalogSignal.

The spatial aspects of the data must be provided with the following keywords
as annotations to the AnalogSignal. For example:
grid_size: (10,10)
spatial_scale: 0.05  # mm
positions: [(0,1), (0,2), (0,3), (0,4), ...]

For now the pipeline assumes equally spaced electrodes on a rectangular grid
(not all positions must be occupied).
