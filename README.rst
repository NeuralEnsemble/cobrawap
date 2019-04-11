==============
WaveScalephant
==============
This repository hosts the collaboration between WP3.2 and WP5.7

Involved members
----------------
Istituto Nazionale di Fisica Nucleare, Rome: Giulia De Bonis, Pier Stanislao Paolucci, Elena Pastorelli, Francesco Simula

Forschungszentrum Jülich: Michael Denker, Robin Gutzen, Alper Yegenoglu

Overarching goal, strategy and roadmap
--------------------------------------
The overall outcome of this collaboration is to focus on the WaveScalES simulation work and data analysis performed in SP3, and make it accessible as a workflow in a collaborative fashion using tools from SP5 (e.g., Elephant), 6 (e.g., NEST), and 7 (e.g., storage).

The goal of WaveScalES_ is to unveil the underlying mechanisms of deep sleep, anaesthesia and coma, the emergence toward wakefulness, and the link between sleep and learning, taking advantage of cortical slow wave activity (SWA) and investigating it with experimental data, analysis tools, modulation techniques, theoretical models and simulations of such states and of the transition to wakefulness.
Sleep is present in all animal species notwithstanding the risk associated with the disconnection from the environment (e.g. predation) and the reduction of time available for food search and reproduction. Indeed, it is known that the human brains need healthy sleep, as chronic sleep deprivation reduces cognitive performances.

In the framework of a collaboration between INFN, ISS, IDIBAPS and the Juelich Elephant team, we delivered a prototype of the Slow Waves Analysis Pipeline (characterisation of the cortex activity during deep sleep and anaesthesia); the preliminary version can be downloaded from this github. 
SWAP is currently a python pipeline based on Elephant, but the plan is to integrate the SWAP pipeline as a module into Elephant and offer it through the HBP platform by May 2020, as part of the HBP offer. The process of data curation and integration in the Knowledge Graph is ongoing.
SWAP can be applied to experimental data and simulation outputs. It has been validated on an extensive in vivo data set, collected from the cerebral cortex of mice by Multi-Electrode Array. SWAP differentiates by area key-parameters related to the onset of slow oscillations. For example, it demonstrates gradients of key observables along the fronto-lateral to occipito-medial direction in recordings of anaesthetised mice. The pipeline discriminates between brain states, specifically different levels of anaesthesia. It also allows comparing simulation outputs obtained with different simulation engines (for example NEST or DPSNN simulations). 
The final release of the SWAP analysis pipeline, integrated in the HBP infrastructure, and the related set of curated examples of experimental data will be offered after the embargo period (May 2020) to the external community. Researchers will then either apply SWAP to their own experimental data or to the analysis of HBP curated data accessible through the HBP Knowledge Graph.

.. _WaveScalES: https://drive.google.com/file/d/1BYZmhz_qJ8MKPOIeyTZw6zjqfVMcCCCk/view

Current status and possible developments
----------------------------------------

This workflow would include as a first step setting up a "scaffold" or prototype upon which to build:

1. Performing the simulation of a WaveScalES model using DPSNN and NEST on an HBP HPC system (e.g., JULIA) from within the notebook;

2. Transformation of the simulation outputs from NEST and DPSNN into a unified and practical Neo data representation;

3. Trivial analysis of the data using existing Elephant analysis functions (here it was mentioned: firing rates, spectra, wave direction,...);

4. Visualization of the output in a Jupyter notebook;

5. Reproducible analysis of electrophysiological data. 


From this ground, several directions of work are anticipated:

* Integration of the analysis into concrete comparisons/validations of the HBP validation framework (currently developed by SP6 and members at Juelich). This could include two types of validations:
    * models for DPSNN against NEST (i.e., are both engines giving comparable output?);
    * different models developed by SP3 (how do different models/parameters/... differ in the activity they produce?). This step would involve work in getting the WaveScalES models into the HBP model catalog.

* Use case for the online analysis of a running simulations.
    * Currently, work has begun in Juelich to develop systems that enable users to couple simulations running on NEST to online visualization and analysis (in collab. with Simlab Neuroscience).
    * This is an ambitious project, but this workflow may provide good use cases:
        * Perform rudimentary analysis of waves and visualize them online;
        * Online visualization of synaptic strengths and evolution of plasticity with time.

* Bring together analysis methods used by both labs for the analysis and characterization of wave-like activity and UP/DN state detection in the Elephant tool.
    * Impact: Validated, common analysis tools; exchange of methods and knowledge across labs;
    * Further work: Joint development of viewing capabilities of wave activity (e.g., functions to efficient plot waves in the collab, movie like wave plotting).

* Extension of scaffold to the comparison of simulated data vs. experimental data.
    * This would be performed formally in a separate collaboration, however, despite new challenges associated with experimental datasets (e.g., register data with the SP5 Knowledgegraph and Neural Activity Resource), there would be very strong syngergies coming from this work.

Repository structure
--------------------

* The submodules *EphysData-Analysis* and *OpticalData-Analysis* contain the code of the orignial analysis pipeline.

* *snakemake_workflow* contains the rewritten Python analysis scripts, a corresponding Python environment, and a workflow description specified with Snakemake_.

    The data on which this analysis is based is stored here_.

.. _here: https://drive.google.com/drive/folders/1A1UDfkWklRYqinyaX8ednXBa2DnK58Lx?usp=sharing

* *showcase_notebooks*

    * *Ephys_sandbox.ipynb* illustrates the individual analysis steps of the snakemake workflow
    * *DPSNN_NEST.ipynb* shows the application of various validation methods to the comparison of the simulator outcomes by the DPSNN and NEST engines.

.. _Snakemake: https://snakemake.readthedocs.io/en/stable/


Snakemake workflow introduction
-------------------------------

* *scripts* folder: contains all the scripts required by the analysis workflow

* *settings.py* specifies the paths to the required scripts and data sets (needs to be individually adjusted!)

* *configfile.yaml* specifies the tunable parameters of the workflow

* *Snakefile* specifies the individual steps of the workflow in the form of separate rules

**How to run the workflow**

navigate to the snakemake folder

.. code:: bash

    cd snakemake_workflow

Snakemake enables to generate various result files and plots along the steps of workflow by asking for the corresponding output file.
The current options are:

.. code:: bash

    snakemake /path/to/../results/161101_rec01_Spontaneous_RH.nix
    snakemake /path/to/../results/logMUA.nix
    snakemake /path/to/../results/UD_state_vector.npy

When generating the figures, parameters can be passed in the filename such as {channel id}, {t_start}, {t_stop}, and {output format}.
For example:

.. code:: bash

    snakemake /path/to/../results/figures/lfp_traces_t280-304s.png
    snakemake /path/to/../results/figures/power_spectrum.pdf
    snakemake /path/to/../results/figures/logMUA_states_channel2_280-282s.png
    snakemake /path/to/../results/figures/UD_slopes_channel5.jpg

In case you haven't set up a Python environment which is able to run the scripts, simply add the flag
:code:`--use-conda` to automatically generate an appropriate conda environment on the fly.
This requires a distribution of conda to be install (e.g. miniconda_).


.. _miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/







