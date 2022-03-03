# Stage 2 - Processing
This stage prepares the data for analysis. The user can select the required processing steps depending on the data and analysis objectives.

[config template](configs/config_template.yaml)
<!-- |
[example report](../../examples/reports/report_stage02-preprocessing.html) -->

#### Input
Simultaneous neural activity recordings from spatially arranged (on a grid) electrodes/pixels

* A neo.Block object containing
an AnalogSignal with all signal channels (additional AnalogSignal objects are ignored) with
    * array_annotations: _'x_coords'_ and _'y_coords'_ specifying the integer position on the electrode/pixel grid of the channels
    * annotation: _'spatial_scale'_ specifying the distance between electrodes/pixels as quantities.Quantity object

#### Output
Activity signals, cleaned and pre-processed to user specifications

* Format and shape is identical to the input. AnalogSignal.description contains a summary of the preformed processing steps

## Usage
In this stage all blocks can be selected and arranged in arbitrary order. The execution order is specified by the config parameter `BLOCK_ORDER`.

Other as in other stages, in this stage, all blocks are re-executed with every snakemake call. Because, the execution order of the blocks is completely modular and might change in-between runs, rerunning all blocks with the latest execution order ensures that no intermediate file with a contradicting origination confounds the results.

## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__frequency_filter__|low/high/bandpass filters signal|`HIGHPASS_FREQ`, `LOWPASS_FREQ`, `FILTER_ORDER`, `FILTER_FUNCTION`, _`PSD_FREQ_RES`_, _`PSD_OVERLAP`_|
|__background_subtraction__|subtracts average of each channel| |
|__spatial_downsampling__|spatial smoothing by factor|`MACRO_PIXEL_DIM`|
|__normalization__|divides signals by factor|`NORMALIZE_BY`|
|__roi_selection__|masks area of low signal intensity|`INTENSITY_THRESHOLD`|
|__detrending__|removes (linear, quadratic, ..) trends in signals|`DETRENDING_ORDER`|
|__hierarchical_spatial_downsampling__|non homogeneous spatial smoothing according to macro pixels sigma to noise ratio|`EXIT_CONDITION`, `SIGNAL_EVALUATION_METHOD`, `N_BAD_NODES`, `VOTING_THRESHOLD`|

(_plotting parameters in italic_)
