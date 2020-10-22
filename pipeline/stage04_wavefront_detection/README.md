# Stage 04 - Wavefront Detection
This stage detects individual propagating waves (wave events - making up a wave collection). Currently, two approaches have been implemented in the pipeline, based on either the detected transition times (triggers): 
1) WaveHunt_PropagationClustering --> a wave event is a cluster of triggers, grouped according given assumptions (spatio-temporal causality) 
2) WaveHunt_TimeCropping --> the time sequence of triggers is opportunely cropped to isolate wave events  

[config template](config_template.yaml)

#### Input
Simultaneous neural activity recordings from spatially arranged (on a grid) electrodes/pixels

A neo.Block object containing
* an AnalogSignal with all signal channels with
    * array_annotations: _'x_coords'_ and _'y_coords'_ specifying the integer position on the electrode/pixel grid of the channels
    * annotation: _'spatial_scale'_ specifying the distance between electrodes/pixels as quantities.Quantity object
* an Event object named _'Transitions'_ with
    * labels: _'UP'_
    * array_annotations: _'x_coords'_ and _'y_coords'_

#### Output
Input signals and transition event + wavefronts as collections of transitions times as an Event object in the same neo.Block

* AnalogSignal is identical to the input
* (additional AnalogSignal _'Optical Flow'_ containing the velocity vector field as complex valued signals)
* neo.Event object named _'Wavefronts'_
    * labels: wavefront id
    * annotations: parameters of clustering algorithm, copy of transitions event annotations
    * array_annotations: _'channels'_, _'x_coords'_, _'y_coords'_


## Usage
In this stage two possible algorithms can be used. The algorithm to be used is specified by the config parameter `DETECTION_BLOCK`.


## Blocks
|Name | Description | Parameters |
|:----|:------------|:-----------|

———> Blocks concerning the WaveHunt_clustering algorithm:
|__clustering__|groups trigger events by spatial and temporal distance|`METRIC`, `NEIGHBOUR_DISTANCE`, `MIN_SAMPLES_PER_WAVE`, `TIME_SPACE_RATIO`|
|__(optical_flow)__|calculates vector velocity field with Horn-Schunck algorithm|`ALPHA`, `MAX_NITER`, `CONVERGENCE_LIMIT`, `GAUSSIAN_SIGMA`, `DERIVATIVE_FILTER`|
|__(critical_points)__|..|..|

———> Blocks concerning the WaveHunt_Time algorithm:
|__WaveHunt__|splitting the set of transition times into separate waves according to the unicity principle (i.e. every channel cannot be involved more than once by the passage of a single wave) and a globality principle (i.e. we only keep waves for which at least the 75\% of the total pixels are recruited). This selection is done in order to guarantee that what we call “wave” is actually a global collective phenomenon on the cortex.  | ‘DIM_X’, ‘DIM_Y’, ‘PIXEL_SIZE’, ‘MACRO_PIXEL_DIM’, ‘SPATIAL_SCALE’, ‘Max_Abs_Timelag’, ‘Acceptable_rejection_rate’, ‘meanUP’, ‘ThR1’, ‘ThR2’, ‘MIN_CH_NUM’
