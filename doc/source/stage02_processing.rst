
.. include:: ../../cobrawap/pipeline/stage02_processing/README.rst

Blocks
======

.. currentmodule:: stage02_processing.scripts

Utility Blocks (*fixed*)
------------------------
.. autosummary::
   :toctree: _toctree/stage02_processing/
   :template: block

    check_input
    plot_processed_trace

Processing Blocks (*choose any*)
--------------------------------
.. autosummary::
   :toctree: _toctree/stage02_processing/
   :template: block

    background_subtraction
    detrending
    frequency_filter
    logMUA_estimation
    normalization
    phase_transform
    roi_selection
    spatial_downsampling
    subsampling
    zscore
