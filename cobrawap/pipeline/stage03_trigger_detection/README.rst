============================
Stage 03 - Trigger Detection
============================

**This stage detects the potential passing of wavefronts on each channel (for example, transitions between Down and Up states) as trigger times.**

`config template <https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage03_trigger_detection/configs/config_template.yaml>`_

Input
=====

A ``neo.Block`` and ``Segment`` object containing an ``AnalogSignal`` object containing all signal channels (additional ``AnalogSignal`` objects are ignored).

*should pass* |check_input|_

.. |check_input| replace:: *check_input.py*
.. _check_input: https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage03_trigger_detection/scripts/check_input.py

Output
======

The same input data object, but extended with a ``neo.Event`` object named *'transitions'*, containing

* *times*: time stamps where a potential wavefront, i.e., state transition, was detected,
* *labels*: either ``UP`` or ``DOWN``,
* *annotations*: information about the detection methods and copy of ``AnalogSignal.annotations``,
* *array_annotations*: ``channels`` and the ``array_annotations`` of the ``AnalogSignal`` object that correspond to the respective channels.

The output ``neo.Block`` is stored in ``{output_path}/{profile}/stage03_trigger_detection/trigger_times.{NEO_FORMAT}``.

The intermediate results and plots of each processing block are stored in the ``{output_path}/{profile}/stage03_trigger_detection/{block_name}/``.

Usage
=====
In this stage offers alternative trigger detection methods (*choose one*), which can be selected via the ``DETECTION_BLOCK`` parameter.
There are additional filter blocks to post-process the detected triggers, they can be selected (*choose any*) via the ``TRIGGER_FILTER`` parameter.
