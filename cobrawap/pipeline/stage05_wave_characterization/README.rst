================================
Stage 05 - Wave Characterization
================================

**This stage evaluates the detected waves by deriving characteristic wave-wise measures.**

`config template <https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage05_wave_characterization/configs/config_template.yaml>`_

Input
=====
A ``neo.Block`` and ``Segment`` object containing

a ``neo.Event`` object named _'wavefronts'_, containing

* *labels*: wave ids,
* *array_annotations*: ``channels``, ``x_coords``, ``y_coords``.

* Some blocks may require the additional ``AnalogSignal`` object called *'optical_flow'* but containing the complex-valued optical flow values.

*should pass* |check_input|_

.. |check_input| replace:: *check_input.py*
.. _check_input: https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage05_wave_characterization/scripts/check_input.py

Output
======
A table (``pandas.DataFrame``), containing

* the wave-wise characteristic measures, their unit, and if applicable their uncertainty as determined by the selected blocks
* any annotations as selected via ``INCLUDE_KEYS`` or ``IGNORE_KEYS``

Usage
=====
In this stage, any number of blocks can be selected via the ``MEASURES`` parameter and are applied on the stage input (*choose any*).
To include specific metadata in the output table, select the corresponding annotation keys with ``INCLUDE_KEYS``, or to include all available metadata execept some specifiy only the corresponding annotations keys in ``IGNORE_KEYS``.
