# Stage 03 - Trigger Detection

**This stage detects the potential passing of wavefronts on each channel (for example, transitions between Down and Up states) as trigger times.**

[config template](configs/config_template.yaml)

### Input

* A `neo.Block` and `Segment` object containing an `AnalogSignal` object containing all signal channels (additional `AnalogSignal` objects are ignored).

[should pass [_`check_input.py`_](scripts/check_input.py)]

### Output

* The same input data object, but extended with a `neo.Event` object named _'transitions'_, containing
    * times: time stamps where a potential wavefront, i.e., state transition, was detected,
    * labels: either _'UP'_ or _'DOWN'_,
    * annotations: information about the detection methods and copy of `AnalogSignal.annotations`,
    * array_annotations: _'channels'_ and `AnalogSignal.array_annotations` corresponding to the respective channels.
* The output `neo.Block` is stored in _`{output_path}/{profile}/stage03_trigger_detection/trigger_times.{NEO_FORMAT}`_
* The intermediate results and plots of each processing block are stored in the _`{output_path}/{profile}/stage03_trigger_detection/{block_name}/`_

### Usage
In this stage offers alternative trigger detection methods (_choose one_), which can be selected via the `DETECTION_BLOCK` parameter.
There are additional filter blocks to post-process the detected triggers, they can be selected (_choose any_) via the `TRIGGER_FILTER` parameter.

<!-- |Name | Description | Parameters |
|:----|:------------|:-----------|
|__threshold__|thresholds UP states in channels|`THRESHOLD_METHOD`|
|__calc_threshold_fixed__|calculates values for threshold block|`FIXED_THRESHOLD`|
|__calc_threshold_fitted__|calculates values for threshold block|`FIT_FUNCTION`, `BIN_NUM`, `SIGMA_FACTOR`|
|__minima__|detects UP transitions as local minima. |`NUM_INTERPOLATION_POINTS`, `USE_QUADRATIC_INTERPOLATION`, `MIN_PEAK_DISTANCE`, `MINIMA_PERSISTENCE`, `MAXIMA_THRESHOLD_FRACTION`,  `MAXIMA_THRESHOLD_WINDOW`|
|__hilbert_phase__|detects UP transitions as phase transitions|`TRANSITION_PHASE`|
|__remove_short_states__|removes short UP and/or DOWN states|`MIN_UP_DURATION`, `MIN_DOWN_DURATION`, `REMOVE_DOWN_FIRST`| -->
