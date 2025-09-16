=========================
Stage 04 - Wave Detection
=========================

**This stage detects individual propagating waves based on the local transition times and optionally complements the wave description with additionally derived properties.**

`config template <https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage04_wave_detection/configs/config_template.yaml>`_

Input
=====
A ``neo.Block`` and ``Segment`` object containing

an ``AnalogSignal`` object with all signal channels with

* ``array_annotations``: ``x_coords`` and ``y_coords`` specifying the integer position on the channel grid;

an ``Event`` object named *'transitions'* with

* *times*: time stamps where a potential wavefront, i.e., state transition, was detected,
* *labels*: ``UP`` (``DOWN`` or other are ignored),
* *array_annotations*: ``channels``, ``x_coords``, ``y_coords``

*should pass* |check_input|_

.. |check_input| replace:: *check_input.py*
.. _check_input: https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage04_wave_detection/scripts/check_input.py

Output
======
The same input data object, but extended with a ``neo.Event`` object named *'wavefronts'*, containing

* *times*: ``UP`` transitions times from 'transitions' event,
* *labels*: wave ids,
* *annotations*: parameters of clustering algorithm, copy of transitions event annotations,
* *array_annotations*: ``channels``, ``x_coords``, ``y_coords``

eventually additional ``AnalogSignal`` and ``Event`` objects from the blocks specified as ``ADDITIONAL_PROPERTIES``

* such as an ``AnalogSignal`` object called *'optical_flow'* equivalent to the primary ``AnalogSignal`` object, but containing the complex-valued optical flow values.

The output ``neo.Block`` is stored in ``{output_path}/{profile}/stage04_wave_detection/waves.{NEO_FORMAT}``

The intermediate results and plots of each processing block are stored in the ``{output_path}/{profile}/stage04_wave_detection/{block_name}/``

Usage
=====
In this stage offers alternative wave detection methods (*choose one*), which can be selected via the ``DETECTION_BLOCK`` parameter.
There are blocks to add additional properties, to be selected (*choose any*) via the ``ADDITIONAL_PROPERTIES`` parameter.
