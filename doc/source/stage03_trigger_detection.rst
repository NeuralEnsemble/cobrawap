
.. include:: ../../cobrawap/pipeline/stage03_trigger_detection/README.rst

Blocks
======

.. currentmodule:: stage03_trigger_detection.scripts

Utility Blocks (*fixed*)
------------------------
.. autosummary:: 
   :toctree: _toctree/stage03_trigger_detection/
   :template: block

    check_input
    plot_trigger_times

Detection Blocks (*choose one*)
-------------------------------
.. autosummary:: 
   :toctree: _toctree/stage03_trigger_detection/
   :template: block

    hilbert_phase
    minima
    threshold

Trigger Filter Blocks (*choose any*)
------------------------------------
.. autosummary:: 
   :toctree: _toctree/stage03_trigger_detection/

    remove_short_states
