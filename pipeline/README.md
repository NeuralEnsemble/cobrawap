# Modular Wave Analysis Pipeline
The design of the pipeline aims at interfacing a variety of general and specific analysis and processing steps in a flexible modular manner. Hence, the pipeline is able to adapt to diverse types of data (e.g., electrical ECoG, or optical Calcium Imaging recordings) and to different analysis questions. This makes the analyses a) more reproducible and b) comparable among each other since they rely on the same stack of algorithms and any differences in the processing are fully transparent.
The individual processing and analysis steps, __Blocks__, are organized in sequential __Stages__. Following along the stages, the analysis becomes more specific but also allows to branch off at after any stage, as each stage yields useful intermediate results and is autonomous so that it can be reused and recombined. Within each stage, there is a collection of blocks from which the user can select and arrange the analysis via a config file. Thus, the pipeline can be thought of as a curated database of methods on which an analysis can be constructed by drawing a path along the blocks and stages.

## Basic Structure
* __Snakefile__ defines how the stages are executed within the full pipeline
* __configs__
    * __config.yaml__ defines the global parameter settings
* __settings.py__ defines the `output_path`
* __scripts/__ contains general utility scripts
* __\<stageXY\>/__
  * __Snakefile__ defines the blocks (=rules) and organizes their interaction
  * __scripts/__ contains the analysis and plotting scripts (~1 script per block)
  * __configs/__ contains the parameter settings for this stage
    * __config_\<PROFILE\>.yaml__ with different settings for different profiles

## Configuration
All config files are given as templates. So, in order to get started you need to copy (and edit to your needs)
_config_template.yaml_ to _config.yaml_ in the _pipeline/_ folder.
Similarly, _settings_template.py_ needs to be copied to _setting.py_ and the containing `output_path` set fit your local system.

To organize configurations for different datasets or applications over all stages, you can specify profiles. The `PROFILE` parameter in the pipeline config file selects the stage config files (*\<stage\>/configs/config_\<PROFILE\>.yaml*). The results of different profiles are also stored in separate locations (*output_path/\<PROFILE\>/...*)

[see pipeline config](configs/config_template.yaml)

## Execution
The required Python packages are defined in the _environment.yaml_ file. We suggest to use [conda](https://docs.conda.io/en/latest/) for the environment management.

```
conda env create --file environment.yaml
conda activate mowap
```

#### Full Pipeline
Navigate to the _pipeline/configs_ folder. The _config.yaml_ file defines the global parameters.
Most importantly here, the `STAGES` parameter defines which stages are executed, and the `PROFILE` parameter sets which configs files are used in the stages.
For parameters that are defined in the global and the stage configs, the global configs have priority. So, parameters, e.g. for plotting, can be set for all stages.
Once the configurations are handled the execution is a simple snakemake call:

`snakemake --cores=1`

Optionally, parameters can also be directly set via the command line (e.g to quickly change profiles):

`snakemake --config PROFILE=<PROFILE> --cores=1`

#### Single Stage
Navigate to the stage folder. As each stage is a subworkflow it can be executed with the same snakemake calls as the full pipline. Only the config file needs to be explicitly specified:

`snakemake --configfile='configs/config_<PROFILE>.yaml' --cores=1`

In case the `STAGE_INPUT` file is not found, it needs to be set manually either by adding the full path to the config file or via the command line:

`snakemake --configfile='configs/config_<PROFILE>.yaml' --config STAGE_INPUT=/path/to/input/file --cores=1`

#### Single Blocks
Each block is represented by a snakemake rule. To run a specific rule you can request its output file (for multiple output files any one will do):

`snakemake /path/to/specific/file --configfile='configs/config_<PROFILE>.yaml' --cores=1`

However, keep in mind that snakemake keeps track of the timestamps of scripts, in- and output files. So, a rule will only be run again if any of its inputs has changed, and if something in the creation of the input changed this might trigger also other rules to be re-executed.

_See the snakemake [documentation](https://snakemake.readthedocs.io/en/stable/executing/cli.html) for additional command line arguments_

## Interfaces
#### Stage inputs
The path to the input file for each stage is defined in the config parameter `STAGE_INPUT`. When executing the full pipeline the stage inputs are automatically set to the outputs of the previous stage, respectively.
Details on the input requirements for each stage are specified in the corresponding Readme.

#### Stage outputs
The stage output file is stored as _output_path/PROFILE/STAGE_NAME/STAGE_OUTPUT_, with `PROFILE`, `STAGE_NAME`, and `STAGE_OUTPUT` taken from the corresponding config file and `output_path` from *settings.py*.
Details on the output content and format for each stage are specified in the corresponding Readme.

#### Block inputs
Input dependencies to blocks are handled by the corresponding rule in the *Snakefile* and are arranged according on the mechanics of the respective stage.

#### Block outputs
All output from blocks (data and figures) is stored hierarchically in _output_path/PROFILE/STAGE_NAME/\<block name\>/_.

## Reports
[*currently disabled because it creates performance issues on clusters*]

Reports are summaries (html page) about the execution of a Snakefile containing the rule execution order, run-time statistics, parameter configurations, and all plotting outputs tagged with `report()` in the Snakefile.

When the whole pipeline is executed, the reports for each stage are automatically created in _output_path/PROFILE/STAGE_NAME/report.html_.
To create a report for an individual stage, you can use the `report` flag.
`snakemake --configfile='configs/config_XY.yaml' --report /path/to/report.html`

Note that when using the option of setting `PLOT_CHANNELS` to `None` to plot a random channel, the report function might request a different plot than was previously created and will thus fail.
