=====================
Stage 01 - Data Entry
=====================

**This stage handles the loading and representation of the input dataset and metadata into the standard format for the pipeline.**

`config template <https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage01_data_entry/configs/config_template.yaml>`_

Input
=====
A dataset containing neural activity signals simultaneously recorded from channels (electrodes/pixels) laying on a grid. This includes data from various signal modalities from various measurement techniques or simulations. See required `data capabilities <#required-data-capabilities>`_ and `metadata <#required-metadata>`_ below.

Furthermore it requires a (custom) loading script to load the dataset and bring it into the required representation. See `guide to create loading script from a template <#entering-datasets-into-cobrawap>`_ below.

Output
======
The input data and metadata represented in the Neo_ format. Concretely, a ``neo.Block`` and ``Segment`` object containing an ``AnalogSignal`` object containing all signal channels (additional ``AnalogSignal`` objects are ignored) with at least

.. _Neo: https://neo.readthedocs.io/

* *array_annotations*: ``x_coords`` and ``y_coords`` specifying the integer position on the channel grid;
* *annotations*: ``spatial_scale`` specifying the distance between electrodes/pixels as ``quantities.Quantity`` object.

Any additional metadata and neo objects in the ``neo.Block`` passed along through the pipeline and may complement the final pipeline output.

*should pass* |check_input|_

.. |check_input| replace:: *check_input.py*
.. _check_input: https://github.com/NeuralEnsemble/cobrawap/blob/master/pipeline/stage01_data_entry/configs/scripts/check_input.py

Required Data Capabilities
==========================
*What kind of data can go into the pipeline?*

* It needs to exhibit propagating wave-like activity, for example, in the form of spatially organized transitions between Up and Down states (slow waves) or spatially phase-shifted field potential oscillations.
* Channels (electrodes/pixels) must be regularly spaced on a rectangular grid, which can include empty sites.

Required Metadata
=================
* Minimum metadata (*required, for a correct processing of the data*)
   * Sampling rate of the signals (set as attribute of ``AnalogSignal``)
   * Distance between channels (``quantities.Quantity`` object set as annotation ``spatial_scale`` in ``AnalogSignal``)
   * Relative spatial location of channels (``int`` arrays as array_annotations ``x_coords`` and ``y_coords`` in ``AnalogSignal``)

* Recommended metadata (*desired, for a correct interpretation of the results*)
    * Identifier/source of the dataset
    * Units of AnalogSignal
    * Anatomical orientation of the recorded region
    * Absolute cortical positioning of the electrodes
    * Type and dosage (or estimated level) of anesthetic
    * Species and general animal information
    * Information on artifacts and erroneous signals
    * Any additional protocols or events (e.g. stimulation) influencing the signals

Entering Datasets Into Cobrawap
===============================
Datasets can be very different. Therefore, it usually requires a custom loading script to access the data and bring it into a standard representation that can be used by the pipeline.
Having the data and metadata already in a standard format (e.g., Neo_) as used within the pipeline) makes this step easier. Similar datasets which, for example, come from the same experiment may be able to use the same loading script.

1. Put the dataset somewhere accessible.
    It should be accessible from where the pipeline is intended to run (i.e. local machine, compute cluster, ...)

2. Create the corresponding config file.
    * Rename the ``config_template.yaml`` in ``stage01_data_entry/configs/`` to ``config_<data-name>.yaml``. This can be either within the ``cobrawap/pipeline`` folder or your ``<configs_dir>/`` specified in ``settings.py``.
    * Set the path to the dataset for ``DATA_SETS`` as ``<data-name>: '/path/to/data/'``.
    * Set the name of the loading script for ``CURATION_SCRIPT`` (*will be created in the next step*).
    * Set the additional metadata for the config parameter as required (*can be revisited when writing the loading script*)

3. Create the corresponding loading script.
    * Copy the ``enter_data_template.py`` in ``cobrawap/pipeline/stage01_data_entry/scripts/`` and and name it as specified in the just created config file.
    * Put the script either in the same folder or in ``<configs_dir>/stage01_data_entry/scripts/`` if defined.

4. Edit the loading script.
    * Write a loading routine for your data type. Also, check whether there is a `neo IO function <https://neo.readthedocs.io/en/stable/io.html#module-neo.io>`_ to load the data directly into the neo structure.
    * Follow along the template to create a neo structure with a ``neo.Block``, ``Segment``, and ``AnalogSignal`` containing all signal channels. You may use the utility functions ``merge_analogsignals()`` when each channel is loaded as as separate ``AnalogSignal``, or the function ``imagesequence_to_analogsignal()`` when loading imaging data as an ``ImageSequence`` object.
    * As outlined in the template add the required metadata (and extra metadata) as ``annotations`` and ``array_annotations`` to the ``AnalogSignal`` object. When some metadata is not loaded with the data object, it can be specified in the config file (in ``ANNOTATIONS``, ``ARRAY_ANNOTATIONS``, as well as ``KWARGS``) to be available in the loading script and added manually to the ``AnalogSignal`` object.

5. Test and debug the loading script.
    * Follow the instructions in the `pipeline README <../README.md>`_ to execute the pipeline or only stage 1.
    * After the execution of the ``enter_data`` block, the created data object is automatically checked whether it adheres to the pipeline's requirements via the ``check_input`` block.
    * Even when there is no error in the execution, inspect the results for correctness in ``<output_path>/stage01_data_entry/``.
