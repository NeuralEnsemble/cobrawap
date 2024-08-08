*************
Release Notes
*************


Release 0.2.0
=============
New functionality and features
------------------------------
* Ability to plot complete signal ranges using `TSTART` and `TSTOP` set to `None` (`#48 <https://github.com/NeuralEnsemble/cobrawap/pull/48>`_)
* New default value `None` for `MAXIMA_THRESHOLD_WINDOW` to indicate that the complete signal duration is considered (`#49 <https://github.com/NeuralEnsemble/cobrawap/pull/49>`_)
* Added additional keyword arguments to `cobrawap` command (`#76 <https://github.com/NeuralEnsemble/cobrawap/pull/76>`_)

Bug fixes
---------
* Fixed bug related to updating of AnalogSignal names (`#67 <https://github.com/NeuralEnsemble/cobrawap/pull/67>`_)
* Fixed issue where `roi_selection` and `spatial_derivative` incorrectly handled boolean arguments (`#65 <https://github.com/NeuralEnsemble/cobrawap/pull/65>`_)
* Fixed issue related to directly specifying a stage from the cobrawap interface (`#70 <https://github.com/NeuralEnsemble/cobrawap/pull/70>`_)
* Fixed issue with cyclic boundary conditions during phase convolution (`#66 <https://github.com/NeuralEnsemble/cobrawap/pull/66>`_)

Documentation
-------------
* Updated `README` information (`#59 <https://github.com/NeuralEnsemble/cobrawap/pull/59>`_), (`#74 <https://github.com/NeuralEnsemble/cobrawap/pull/74>`_), (`#77 <https://github.com/NeuralEnsemble/cobrawap/pull/77>`_)

Other changes
-------------
* Automated package distribution to PyPI (`#62 <https://github.com/NeuralEnsemble/cobrawap/pull/62>`_)


Release 0.1.1
=============
Documentation
-------------
* Added help statement for CLI client

Bug fixes
---------
* Fixed install by disallowing Snakemake versions >=8.0.0, which are missing subworkflow support

Selected dependency changes
---------------------------
* snakemake >= 7.10.0, < 8.0.0


Release 0.1.0
=============
Initial release of Cobrawap accompanying the manuscript

Gutzen, R., De Bonis, G., De Luca, C., Pastorelli, E., Capone, C., Allegra Mascaro, A. L., Resta, F., Manasanch, A., Pavone, F. S., Sanchez-Vives, M. V., Mattia, M., Grün, S., Paolucci, P. S., & Denker, M. (2022). *A modular and adaptable analysis pipeline to compare slow cerebral rhythms across heterogeneous datasets*. Cell Reports Methods 4, 100681. `https://doi.org/10.1016/j.crmeth.2023.100681 <https://doi.org/10.1016/j.crmeth.2023.100681>`_


