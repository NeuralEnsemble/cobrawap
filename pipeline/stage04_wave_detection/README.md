# Stage 04 - Wave Detection

**This stage detects individual propagating waves based on the local transition times and optionally complements the wave description with additionally derived properties.**

[config template](configs/config_template.yaml)

#### Input
A `neo.Block` and `Segment` object containing
* an `AnalogSignal` object with all signal channels with
    * `array_annotations`: `x_coords` and `y_coords` specifying the integer position on the channel grid;
* an `Event` object named _'transitions'_ with
    * times: time stamps where a potential wavefront, i.e., state transition, was detected,
    * labels: `UP` (`DOWN` or other are ignored),
    * array_annotations: `channels`, `x_coords`, `y_coords`

#### Output
* The same input data object, but extended with a `neo.Event` object named _'wavefronts'_, containing
    * times: `UP` transitions times from 'transitions' event,
    * labels: wave ids,
    * annotations: parameters of clustering algorithm, copy of transitions event annotations,
    * array_annotations: `channels`, `x_coords`, `y_coords`
* eventually additional `AnalogSignal` and `Event` objects from the blocks specified as `ADDITIONAL_PROPERTIES`
    * an `AnalogSignal` object called *'optical_flow'* equivalent to the primary `AnalogSignal` object, but containing the complex-valued optical flow values.
* The output `neo.Block` is stored in _`{output_path}/{profile}/stage04_wave_detection/waves.{NEO_FORMAT}`_
* The intermediate results and plots of each processing block are stored in the _`{output_path}/{profile}/stage04_wave_detection/{block_name}/`_

## Usage
In this stage offers alternative wave detection methods (_choose one_), which can be selected via the `DETECTION_BLOCK` parameter.
There are blocks to add additional properties, to be selected (_choose any_) via the `ADDITIONAL_PROPERTIES` parameter.


<!-- ## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__clustering__|groups trigger events by spatial and temporal distance|`METRIC`, `NEIGHBOUR_DISTANCE`, `MIN_SAMPLES_PER_WAVE`, `TIME_SPACE_RATIO`|
|__(optical_flow)__|calculates vector velocity field with Horn-Schunck algorithm|`ALPHA`, `MAX_NITER`, `CONVERGENCE_LIMIT`, `GAUSSIAN_SIGMA`, `DERIVATIVE_FILTER`|
|__(critical_points)__|..|..| -->
