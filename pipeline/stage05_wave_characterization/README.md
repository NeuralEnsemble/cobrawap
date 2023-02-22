# Stage 05 - Wave Characterization

**This stage evaluates the detected waves by deriving characteristic wave-wise measures.**

[config template](configs/config_template.yaml)

#### Input
A `neo.Block` and `Segment` object containing
* a `neo.Event` object named _'wavefronts'_, containing
    * labels: wave ids,
    * array_annotations: `channels`, `x_coords`, `y_coords`.
* Some blocks may require the additional `AnalogSignal` object called *'optical_flow'* but containing the complex-valued optical flow values.

#### Output
A table (`pandas.DataFrame`), containing
* the wave-wise characteristic measures, their unit, and if applicable their uncertainty as determined by the selected blocks
* any annotations as selected via `INCLUDE_KEYS` or `IGNORE_KEYS`

## Usage
In this stage, any number of blocks can be selected via the `MEASURES` parameter and are applied on the stage input (_choose any_). 
To include specific metadata in the output table, select the corresponding annotation keys with `INCLUDE_KEYS`, or to include all available metadata execept some specifiy only the corresponding annotations keys in `IGNORE_KEYS`. 

<!-- 
## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__direction__|interpolates directions of planar waves||
|__velocity_planar__|interpolates planer propagation velocity|| -->
