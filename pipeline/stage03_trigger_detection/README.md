# Stage 03 - Trigger Detection
This stage detects the transitions from DOWN to UP states (and back, if applicable) as trigger times.

[config template](config_template.yaml)

#### Input
Signals containing distinguishable states of low and high activity

* A neo.Block object containing an AnalogSignal with all signal channels

#### Output
Input signals + the Up (and Down) trigger times in for each channel as an Event object in the same neo.Block

* AnalogSignal is identical to the input
* neo.Event object named _'Transitions'_
    * labels: either _'UP'_ or _'DOWN'_
    * annotations: information about the detection methods, copy of AnalogSignal.annotations
    * array_annotations: _'channels'_, AnalogSignal.array_annotations of the respective channel

__Plots__
...

## Usage
In this stage offers alternative trigger detection methods, from which one can be selected via the `DETECTION_BLOCK` parameter.

## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|
|__threshold__|thresholds UP states in channels|`THRESHOLD_METHOD`|
|__calc_threshold_fixed__|calculates values for threshold block|`FIXED_THRESHOLD`|
|__calc_threshold_fitted__|calculates values for threshold block|`FIT_FUNCTION`, `BIN_NUM`, `SIGMA_FACTOR`|
|__minima__|detects UP transitions as local minima|`MINIMA_ORDER`|
|__hilbert_phase__|detects UP transitions as phase transitions|`TRANSITION_PHASE`|
