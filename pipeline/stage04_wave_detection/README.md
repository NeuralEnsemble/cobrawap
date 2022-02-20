# Stage 04 - Wave Detection
This stage detects individual propagating waves based on either the detected transition times (or the input signals directly).

[config template](configs/config_template.yaml)

#### Input
Simultaneous neural activity recordings from spatially arranged (on a grid) electrodes/pixels

A neo.Block object containing
* an AnalogSignal with all signal channels with
    * array_annotations: _'x_coords'_ and _'y_coords'_ specifying the integer position on the electrode/pixel grid of the channels
    * annotation: _'spatial_scale'_ specifying the distance between electrodes/pixels as quantities.Quantity object
* an Event object named _'transitions'_ with
    * labels: _'UP'_
    * array_annotations: _'x_coords'_ and _'y_coords'_

#### Output
Input signals and transition event + wavefronts as collections of transitions times as an Event object in the same neo.Block

* AnalogSignal is identical to the input
* (additional AnalogSignal _'optical_flow'_ containing the velocity vector field as complex valued signals)
* neo.Event object named _'wavefronts'_
    * labels: wavefront id
    * annotations: parameters of clustering algorithm, copy of transitions event annotations
    * array_annotations: _'channels'_, _'x_coords'_, _'y_coords'_

## Usage
..

## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__clustering__|groups trigger events by spatial and temporal distance|`METRIC`, `NEIGHBOUR_DISTANCE`, `MIN_SAMPLES_PER_WAVE`, `TIME_SPACE_RATIO`|
|__(optical_flow)__|calculates vector velocity field with Horn-Schunck algorithm|`ALPHA`, `MAX_NITER`, `CONVERGENCE_LIMIT`, `GAUSSIAN_SIGMA`, `DERIVATIVE_FILTER`|
|__(critical_points)__|..|..|
