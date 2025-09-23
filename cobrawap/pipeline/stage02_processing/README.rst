=====================
Stage 02 - Processing
=====================
**This stage prepares the data for analysis. The user can select the required processing steps depending on the data and analysis objectives.**

`config template <https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage02_processing/configs/config_template.yaml>`_

Input
=====
Simultaneous neural activity recordings from electrodes/pixels, spatially arranged on a grid.

A ``neo.Block`` and ``Segment`` object containing an ``AnalogSignal`` object containing all signal channels (additional ``AnalogSignal`` objects are ignored) with

* *array_annotations*: ``x_coords`` and ``y_coords`` specifying the integer position on the channel grid;
* *annotations*: ``spatial_scale`` specifying the distance between electrodes/pixels as ``quantities.Quantity`` object.

*should pass* |check_input|_

.. |check_input| replace:: *check_input.py*
.. _check_input: https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage02_processing/scripts/check_input.py

Output
======
* The same structured ``neo.Block`` object containing an ``AnalogSignal`` object. The channel signals in ``AnalogSignal`` are processed by the specified blocks and parameters.
* The respective block parameters are added as metadata to the annotations of the ``AnalogSignal``.
* The output ``neo.Block`` is stored in ``{output_path}/{profile}/stage02_processing/processed_data.{NEO_FORMAT}``
* The intermediate results and plots of each processing block are stored in the ``{output_path}/{profile}/stage02_processing/{block_name}/``

Usage
=====
In this stage, all blocks can be selected and arranged in arbitrary order (*choose any*). The execution order is specified by the config parameter ``BLOCK_ORDER``. All blocks, generally, have the same output data representation as their input, just transforming the ``AnalogSignal`` and adding metadata, without adding data objects.

When the block order is changed in-between runs, it may happen that not all the necessary blocks are re-executed correctly, because of Snakemake's time-stamp-based re-execution mechanism. Therefore, to be sure all blocks are re-executed, you can set ``RERUN_MODE`` is set to ``True``. However, when you are not changing the block order, setting it to ``False`` prevents unnecessary reruns.
