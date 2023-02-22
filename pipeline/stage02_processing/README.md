# Stage 2 - Processing
**This stage prepares the data for analysis. The user can select the required processing steps depending on the data and analysis objectives.**

[config template](configs/config_template.yaml)

### Input
Simultaneous neural activity recordings from electrodes/pixels, spatially arranged on a grid.

* A `neo.Block` and `Segment` object containing an `AnalogSignal` object containing all signal channels (additional `AnalogSignal` objects are ignored) with
    * `array_annotations`: `x_coords` and `y_coords` specifying the integer position on the channel grid;
    * `annotations`: `spatial_scale` specifying the distance between electrodes/pixels as `quantities.Quantity` object.

[_`check_input.py`_](scripts/check_input.py)

### Output
* The same structured `neo.Block` object containging an `AnalogSignal` object. The channel signals in `AnalogSignal` are processed by the specified blocks and parameters.
* The respective block parameters are added as metadata to the annotations of the `AnalogSignal`.
* The output `neo.Block` is stored in _`{output_path}/{profile}/stage02_processing/processed_data.{NEO_FORMAT}`_
* The intermediate results and plots of each processing block are stored in the _`{output_path}/{profile}/stage02_processing/{block_name}/`_

### Usage
In this stage, all blocks can be selected and arranged in arbitrary order (_choose any_). The execution order is specified by the config parameter `BLOCK_ORDER`. All blocks, generally, have the same output data representation as their input, just transforming the `AnalogSignal` and adding metadata, without adding data objects.

When the block order is changed in-between runs, it may happen that not all the necessary blocks are re-executed correctly, because of Snakemake's time-stamp-based re-execution mechanism. Therefore, to be sure all blocks are re-executed, you can set `RERUN_MODE` is set to `True`. However, when you are not changing the block order, setting it to `False` prevents unnecessary reruns.

<!-- 
## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__frequency_filter__|low/high/bandpass filters signal|`HIGHPASS_FREQ`, `LOWPASS_FREQ`, `FILTER_ORDER`, `FILTER_FUNCTION`, _`PSD_FREQ_RES`_, _`PSD_OVERLAP`_|
|__background_subtraction__|subtracts average of each channel| |
|__spatial_downsampling__|spatial smoothing by factor|`MACRO_PIXEL_DIM`|
|__normalization__|divides signals by factor|`NORMALIZE_BY`|
|__roi_selection__|masks area of low signal intensity|`INTENSITY_THRESHOLD`|
|__detrending__|removes (linear, quadratic, ..) trends in signals|`DETRENDING_ORDER`|

(_plotting parameters in italic_) -->
