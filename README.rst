==============
WaveScalephant
==============
This repository aims at delivering reusable and modular pipelines for a multi-scale, multi-methodology analysis of slow wave activity, brain states and their complexity; the repository hosts the collaboration originally started by HBP-SGA2-(WP3.2 and WP5.7) and extended to other HBP members and partners.

Involved members
----------------
- Istituto Nazionale di Fisica Nucleare (INFN), Roma, Italy: Giulia De Bonis, Pier Stanislao Paolucci, Elena Pastorelli, Francesco Simula.

- Forschungszentrum Jülich, Germany: Michael Denker, Robin Gutzen, Alper Yegenoglu.

- Istituto Superiore di Sanità (ISS), Roma, Italy: Maurizio Mattia, Antonio Pazienti.

- Institut d’Investigacions Biomediques August Pi i Sunyer (IDIBAPS), Barcelona, Spain: Miguel Dasilva, Maria V. Sanchez-Vives.

- European Laboratory for Non-Linear Spectroscopy (LENS), Firenze, Italy: Anna Letizia Allegra Mascaro, Francesco Resta, Francesco Pavone.

- University of Milano (UniMi), Italy: Andrea Pigorini, Thierry Nieus, Marcello Massimini 

License
-------
The wavescalephant project is open source software and is licensed under the GNU General Public License v3 or later.

Citation
--------
Please cite this repository if you use it in your work.

Releases planned by 2020-03-31 (HBP-SGA2-M24)
---------------------------------------------
- **Component C2051** (SOAP r1 - Slow Oscillation Analysis Pipeline). Snakemake integration of the Slow Wave Analyisis Pipleine COmponent cabable of extracting the local features of oscillations, a necessary prerequisite for the analysis of slow waves performed at multi-area level by the SWAP analyis. See RelatedRepositories.rst

- **Component C2053** (SWAP r1 - Slow Wave Analysis Pipeline). Snakemake workflow for a modulare slow wave analyisis pipeline that cna be applied to both optical calcium imaging recordings (GECI technique) and multi-electrode recorsings (ECoG) in mouse. 

Background
----------
Starting point: algorythms described in [De Bonis et al (2019)](https://doi.org/10.3389/fnsys.2019.00070) and in [Celotto et al (2020)](https://www.mdpi.com/629916).

.. _arXiv:1902.08599: https://arxiv.org/abs/1902.08599
.. _arXiv:1811.11687: https://arxiv.org/abs/1811.11687

Overarching goal, strategy and roadmap
--------------------------------------
The overall outcome of this collaboration is to focus on the WaveScalES simulation work and data analysis performed in SP3, and make it accessible as a workflow in a collaborative fashion using tools from SP5 (e.g., Elephant), SP6 (e.g., NEST), and SP7 (e.g., storage).

The goal of WaveScalES_ (description at the date 2019-03-31, SGA2-M12) is to unveil the underlying mechanisms of deep sleep, anaesthesia and coma, the emergence toward wakefulness, and the link between sleep and learning, taking advantage of cortical slow wave activity (SWA) and investigating it with experimental data, analysis tools, modulation techniques, theoretical models and simulations of such states and of the transition to wakefulness.
Sleep is present in all animal species notwithstanding the risk associated with the disconnection from the environment (e.g. predation) and the reduction of time available for food search and reproduction. Indeed, it is known that the human brains need healthy sleep, as chronic sleep deprivation reduces cognitive performances.

In the framework of a collaboration between INFN, ISS, IDIBAPS and the Juelich Elephant team, we delivered a prototype of the Slow Waves Analysis Pipeline (characterisation of the cortex activity during deep sleep and anaesthesia); the preliminary version can be downloaded from this github.
SWAP is currently a python pipeline based on Elephant, but the plan is to integrate the SWAP pipeline as a module into Elephant and offer it through the HBP platform by May 2020, as part of the HBP offer. The process of data curation and integration in the Knowledge Graph is ongoing.
SWAP can be applied to experimental data and simulation outputs. It has been validated on an extensive in vivo data set, collected from the cerebral cortex of mice by Multi-Electrode Array. SWAP differentiates by area key-parameters related to the onset of slow oscillations. For example, it demonstrates gradients of key observables along the fronto-lateral to occipito-medial direction in recordings of anaesthetised mice. The pipeline discriminates between brain states, specifically different levels of anaesthesia. It also allows comparing simulation outputs obtained with different simulation engines (for example NEST or DPSNN simulations).
The final release of the SWAP analysis pipeline, integrated in the HBP infrastructure, and the related set of curated examples of experimental data will be offered to the external community through EBRAINS. Researchers will then either apply SWAP to their own experimental data or to the analysis of HBP curated data accessible through the HBP Knowledge Graph.

.. _WaveScalES: https://drive.google.com/file/d/1BYZmhz_qJ8MKPOIeyTZw6zjqfVMcCCCk/view

WavescalEphant short- and mid- term goals 
-----------------------------------------
* Bring together analysis methods used by different labs for the analysis and characterization of wave-like activity and UP/DN state detection, exploiting existing tools like Elephant.
    * Impact: deliber validated, common analysis tools; exchange of methods and knowledge across labs;
    * Further work: Joint development of viewing capabilities of wave activity (e.g., functions to efficient plot waves in the collab, movie like wave plotting, etc...).

* Extension to the comparison to simulated data vs. experimental data; include also synergies with EBRAINS and other HBP platform resources (e.g. the Knowledge Graph). 


Current status of the project and possible developments
-------------------------------------------------------

The current prototype includes:

1. Reproducible analysis of electrophysiological and optical data, on the characterization of cortical slow wave activity and local slow oscillations. 
2. Integration of existing general tools (Elephant, Neo, Snakemake) and custom implementations (see RelatedRepositories.rtf).
3. Definition of a unique data representation in the framework of Neo and link with the data curation needs.
4. Delivery of showcase Jupyter Notebooks for testbench applications and the visualization of the output.

Possible developments and work in progress relate to:

1. Extension to the output of simulations, aiming at comparing different engines (DPSNN and NEST) and different approaches (spiking vs mean-field); provide notebooks for steering the workflow.
2. Transformation of the simulation outputs into a unified and practical Neo data representation;

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
