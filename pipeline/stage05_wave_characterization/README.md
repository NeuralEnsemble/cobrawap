# Stage 05 - Wave Characterization
This stage evaluates the detected waves by deriving characteristic measures from their dynamics.

[config template](configs/config_template.yaml)

#### Input
Collection of waves in form of groups of trigger times.
<!-- and/or ii) a vector field signal with identified critical points -->

<!-- i) -->
neo.Event object named _'wavefronts'_

    * labels: wavefront id
    * annotations: _'spatial_scale'_
    * array_annotations: _'x_coords'_, _'y_coords'_

#### Output
The characteristic measures per wave in table form (pandas.DataFrame) containing the value, unit, and uncertainty. Each measure is represented in an individual dataframe as well as in a combined dataframe of all measures.

## Usage
Select which characterization measures should be calculated via the `MEASURES` parameter.

## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__direction__|interpolates directions of planar waves||
|__velocity_planar__|interpolates planer propagation velocity||
