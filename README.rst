=====================================================
Collaborative Brain Wave Analysis Pipeline (Cobrawap)
=====================================================

.. image:: https://readthedocs.org/projects/cobrawap/badge/?version=latest
   :target: https://cobrawap.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   :align: left

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10198748.svg
  :target: https://doi.org/10.5281/zenodo.10198748
  :alt: DOI Latest Release
  :align: left

.. image:: https://raw.githubusercontent.com/NeuralEnsemble/cobrawap/master/doc/images/cobrawap_logo.png
   :height: 150px
   :alt: Cobrawap Logo
   :align: left

Cobrawap is an adaptable and reusable analysis pipeline for the multi-scale, multi-methodology analysis of cortical wave activity. The pipeline ingests data from heterogeneous sources of spatially organized neuronal activity, such as ECoG or calcium imaging recordings, as well as the outcome of numerical simulations. The pipeline returns statistical measures to quantify the dynamic wave-like activity patterns found in the data.

`Documentation <https://cobrawap.readthedocs.io>`_ | `Publication <https://doi.org/10.1016/j.crmeth.2023.100681>`_ | `Introductory video <https://www.youtube.com/watch?v=1Qf4zIzV9ow&list=PLvAS8zldX4Ci5uG9NsWv5Kl4Zx2UtWQPh&index=13>`_ | `Use-case demo on the EBRAINS Collab <https://wiki.ebrains.eu/bin/view/Collabs/slow-wave-analysis-pipeline/>`_


Concept
=======

.. image:: https://raw.githubusercontent.com/NeuralEnsemble/cobrawap/master/doc/images/cobrawap_pipeline_approach.png
   :height: 300px
   :alt: Schematic Pipeline Approach
   :align: center


For researchers to be able to effectively reproduce results and build on each other's progress, it is important to not only make results openly accessible and to facilitate data sharing, but also to build the analysis workflows in a shareable and reusable manner.

Making analysis scripts available alongside results and datasets is good. What is even better is to design the analysis workflows for specific research questions in such a manner that they are general and flexible enough to be actively reused in further research. In this way, the rigor of the analysis workflow is increased, and the design of the analysis process is simplified for the researcher.

Cobrawap brings together existing analysis methods, tools, and standards, and interfaces them for the analysis and characterization of cortical wave-like activity and UP/DOWN state detections. Cobrawap serves as a space to gather the various data types exhibiting wave-like activity and their various analysis approaches into the same pipeline. Besides generating easily reproducible and curated results, Cobrawap facilitates the rigorous comparison between datasets of different laboratories, studies and measurement modalities. Furthermore, Cobrawap can be used in the context of model validation and benchmarking of analysis methods. Cobrawap may also act as a template for implementing analysis pipelines in other contexts.

**Cobrawap features...**

* a user-friendly command line interface guiding the setup and usage
* a hierarchical and modular pipeline framework based on the Snakemake_ workflow management tool
* reusable method implementations (*stages* and *blocks*) for standalone applications or integration into workflows
* analysis methods for electrophysiological and optical data on the characterization of cortical wave activity and local oscillations
* visualization of the analysis steps and the intermediate results
* intermediate results curated with annotated metadata

.. _Snakemake: https://snakemake.readthedocs.io/en/stable/

For further developments and feature requests refer to the `Github Issues <https://github.com/NeuralEnsemble/cobrawap/issues>`_.


Citation
========
To refer to the Cobrawap software package in publications, please use:

Cobrawap (`doi:10.5281/zenodo.10198748 <https://doi.org/10.5281/zenodo.10198748>`_;
`RRID:SCR_022966 <https://scicrunch.org/resolver/RRID:SCR_022966>`_)

To cite a specific version of Cobrawap please see version-specific DOIs at:

 `doi:10.5281/zenodo.10198748 <https://doi.org/10.5281/zenodo.10198748>`_

To cite Cobrawap, please use:

Robin Gutzen, Giulia De Bonis, Chiara De Luca, Elena Pastorelli, Cristiano Capone, Anna Letizia Allegra Mascaro, Francesco Resta, Arnau Manasanch, Francesco Saverio Pavone, Maria V. Sanchez-Vives, Maurizio Mattia, Sonja Grün, Pier Stanislao Paolucci, Michael Denker (2024), *A modular and adaptable analysis pipeline to compare slow cerebral rhythms across heterogeneous datasets*, Cell Reports Methods,
Volume 4, Issue 1, `https://doi.org/10.1016/j.crmeth.2023.100681 <https://doi.org/10.1016/j.crmeth.2023.100681>`_


License
=======
Cobrawap is open-source software and is licensed under the `GNU General Public License v3 <https://github.com/NeuralEnsemble/cobrawap/blob/master/LICENSE>`_.


The Cobrawap Community
=================
Cobrawap is currently provided as a `tool <https://www.ebrains.eu/tools/cobrawap>`_ of the `EBRAINS <https://www.ebrains.eu>`_ infrastructure and included in the `EBRAINS-Italy <https://www.ebrains-italy.eu/>`_ initiative. Further details on fundings and resources are in the `Acknowledgments <https://github.com/NeuralEnsemble/cobrawap/blob/master/doc/source/acknowledgments.rst>`_ file in the doc folder.

The **Cobrawap Core Team** is in charge of defining the scientific address of the project and taking care of the continuous maintenance and development of the software. It currently includes:

- **Forschungszentrum Jülich, Germany:** Michael Denker
  
- **Istituto Nazionale di Fisica Nucleare (INFN), Roma, Italy:** Giulia De Bonis, Cosimo Lupo, Federico Marmoreo, Pier Stanislao Paolucci
  
- **New York University, NY, USA:** Robin Gutzen
  
The **Cobrawap Support Team** includes people and partners that offer technical support for the integration of the software in a larger framework of interoperable tools; currently, this function is provided by **Unité de Neurosciences, Neuroinformatics Group, CNRS, France** (Andrew Davison), **Athena Research Center, Greece** (Sofia Karvounari, Eleni Mathioulaki).

**Cobrawap Scientific Partners** (past and present) are:  

- **Istituto Superiore di Sanità (ISS), Roma, Italy**
  
- **Institut d’Investigacions Biomediques August Pi i Sunyer (IDIBAPS), Barcelona, Spain**

- **European Laboratory for Non-Linear Spectroscopy (LENS), Firenze, Italy**

- **University of Milano (UniMi), Italy**
  
Other people involved (past and present):
Anna Letizia Allegra Mascaro (LENS), Irene Bernava (INFN), Cristiano Capone (INFN), Alessandra Cardinale (INFN), Miguel Dasilva (IDIBAPS), Chiara De Luca (INFN), Gianluca Gaglioti (UniMi), Arnau Manasanch (IDIBAPS), Marcello Massimini (UniMi), Maurizio Mattia (ISS), Thierry Nieus (UniMi), Francesco S. Pavone (LENS), Andrea Pigorini (UniMi), Francesco Resta (LENS), Maria V. Sanchez-Vives (IDIBAPS). 


Further Context
===============

Software Ecosystem
------------------
The functionality offered by Cobrawap builds on existing software tools and services.

Neo_ improves interoperability between Python tools for analyzing, visualizing, and generating electrophysiology data by providing a common, shared data object model. The Neo data representation provides a hierarchical data and metadata description for a variety of data types including intracellular and extracellular electrophysiology, electrical data with support for multi-electrode, as well as optical recordings. Furthermore, it supports a wide range of neurophysiology file formats to facilitate reading data from most common recording devices.

The Electrophysiology Analysis Toolkit, Elephant_, is an open-source Python library for analysis methods. It focuses on providing fast and reliable implementations for generic analysis functions for spike train data and time series recordings from electrodes. As community centered project, Elephant aims to serve as a common platform for analysis codes from different laboratories, and a consistent and homogeneous analysis framework.

The Neuroscience Information Exchange, NIX_, format is an API and data format to store scientific data and metadata in a combined representation. Its structure is inspired by common types of neuroscience data, and it acts as one of the primary data formats for the Neo data object model.

.. _Neo: http://neuralensemble.org/neo
.. _Elephant: https://python-elephant.org
.. _NIX: http://g-node.github.io/nix

The Human Brain Project and WaveScalES
----------------------
Cobrawap was originally developed in the context the `Human Brain Project <https://www.humanbrainproject.eu>`_, launched as a use-case initiated within the *WaveScalES* sub-project.
Sleep is present in all animal species notwithstanding the risk associated with the disconnection from the environment (e.g. predation) and the reduction of time available for food search and reproduction. Indeed, it is well known that the human brains need healthy sleep, as chronic sleep deprivation reduces cognitive performances. The goal of the WaveScalES sub-project of the `Human Brain Project <https://www.humanbrainproject.eu>`_ was to unveil the underlying mechanisms of deep sleep, anesthesia and coma, the emergence toward wakefulness, and the link between sleep and learning, taking advantage of cortical slow wave activity (SWA) and investigating it with experimental data, analysis tools, modulation techniques, theoretical models, and simulations of such states and of the transition to wakefulness.
