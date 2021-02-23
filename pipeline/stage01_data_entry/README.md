# Stage 01 - Data Curation
This stage handles the loading and transformation of a dataset into the standard format for the pipeline and the annotation with the required metadata.

__INPUT__: A dataset of raw data of any recording modality and any format, along with information on the data acquisition and the experimental context.

__OUTPUT__: A curated dataset in [Neo](https://github.com/INM-6/python-neo) format, saved as a [Nix](https://github.com/G-Node/nix) file, containing one Segment with one AnalogSignal object and the minimal required metadata. _(Details down below)_

__BLOCKS__: Custom data curation specific to the dataset; check of the data format, metadata, and namespace

## Required Data Capabilities
_What kind of data can go into the SWAP pipeline?_
* It needs to exhibit propagating UP and DOWN states
* Electrodes/Pixels must be regularly spaced on a rectangular grid (can include empty sites)

## Required Metadata
#### Minimum metadata for SWAP pipeline
_required, for a correct processing of the data_
* Sampling rate of the signals
* Distance between electrodes/pixels (as annotation `spatial_scale` in AnalogSignal)
* Relative spatial location of channels (as array_annotations `x_coords` and `y_coords` in AnalogSignal; _[use ChannelIndex objects instead, when [#773](https://github.com/NeuralEnsemble/python-neo/issues/773) is fixed]_)
* Anatomical orientation of the recorded region

#### Recommended metadata for SWAP pipeline
_desired, for a correct interpretation of the results_
* Units of AnalogSignal
* Grid size (as annotation `grid_size` in AnalogSignal, given as a list [dimX,dimY])
* Absolute cortical positioning of the electrodes
* Type and dosage (or estimated level) of anesthetic
* Species, and general animal information
* Information on artifacts and erroneous signals
* Any additional protocols or events (e.g. stimulation) influencing the signals
* Lab where the experiment was performed (+ contact person performing the experiment)

<!--
#### Structure of spatial information in Neo used for this pipeline
* All signals are in an AnalogSignal object (times x N channels)
* It is linked to a ChannelIndex object with
    * the same *name* as the AnalogSignal
    * *channel_ids*, an array with ids of 0 to N
    * *index*, an array with ids of 0 to N
    * *coordinates*, an array of tuples of length N -->

## Adding Datasets into the SWA pipeline
There are two options to insert data into the pipeline.

__Option 1__ is loading the raw data and manually adding the minimum amount of metadata as annotations (+ eventual additional information). This is the quick way to get started with the analysis and getting preliminary insight into the dataset.

__Option 2__ is the proper way to enable deep and reproducible insight, but requires more time and effort. This option would be the full extensive description of a dataset using __a)__ standard formats to structure and represent the data (e.g. [Neo](https://neo.readthedocs.io/) or [BIDS](https://bids.neuroimaging.io)), and __b)__ the inclusion of all available metadata describing every aspect of the experiment, represented in a standardized human and machine readable way (e.g. using [odML](https://g-node.github.io/python-odml/) and [odMLtables](https://github.com/INM-6/python-odmltables)) storing and linking it with the data, and __c)__ the storage in an accessible (versioned) repository alongside a concise documentation, a license, and a citation guide. Some of the aspects of option 2 go hand in hand with filing the dataset into the in the [KnowledgeGraph](https://www.humanbrainproject.eu/en/explore-the-brain/search/).

#### Guide for option 1 - loading raw data and manually adding metadata
The data curation stage is very specific and dependent on the type and format of the given dataset. Therefore, the user typically needs to write a custom curation script to prepare the dataset for the entry into the pipeline. Here, we provide a guideline and template for writing such a script and using it as a block in the Data Curation stage. Each type of dataset (e.g. coming from the same experiment) needs to be given an identifying name `<data_name>`, which is used to link it to the corresponding script and config file.

1. Create a custom curation script and config file

Create a script called *‘curate_\<data_name>.py’* in the scripts folder. Create a config file called *‘config.yaml’*. For the config file it is helpful to copy and edit the provided *‘config_template.yaml’* file or an existing config file of another dataset.

2. Loading the raw data _[in script]_

Check whether there is a Neo IO which can be used to load the data directly into the neo structure (https://neo.readthedocs.io/en/stable/io.html#module-neo.io), if not, write a custom loading routine and add the data into a new Neo object.

3. Adding metadata via the config file _[in config and in script]_

The *config.yaml* file needs to define at least the minimum amount of metadata as defined above. For a generic example see *‘config_template.yaml*. This example config file also introduces the Namespace for the different data and metadata attributes and should not be changed, since later stages build on that Namespace.
In the script, the information specified in the config file are added to the data object as annotations. See *‘scripts/curate_template.py’* for a generic example. The script will store the data as a Nix file in the directory defined by `output_path` in *settings.py* in the subfolder *stage01_data_curation/* with the name specified in the config file.

4. Making sure the script can be used by the curation block _[in Snakefile and in script]_

In the standard case the user would not need to edit the Snakefile. However, in case of errors, there if the naming of the parameters passed from the config file to the script is consistent, the right config file is linked, and the output path is set correctly.

5. Running the curation stage Snakefile manually

Once the whole pipeline is configured it should be run directly from the top level, the *‘pipeline/’* folder, by calling the `snakemake` command. However, to check and debug the curation stage, it can be run on its own, by navigating into the *‘stage01_data_curation/’* folder, and calling `snakemake`. Additional to the Nix data file the stage also produces an example plot of the signals and metadata (using the `PLOT_*` parameters in *config.yaml*), as a check whether the loading and annotations worked correctly.
