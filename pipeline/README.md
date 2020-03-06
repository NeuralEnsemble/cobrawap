### Interfaces
__Pipeline input__:

__Stage inputs__:
The path to the input file for each stage is defined in the config parameter `STAGE_INPUT`. When executing the whole pipeline (calling `snakemake` form the `pipeline/` folder) the stage inputs are automatically set to the outputs of the previous stage, respectively.

__Stage outputs__:
The stage output file is stored as `output_path/STAGE_NAME/STAGE_OUTPUT`, with `STAGE_NAME`, and `STAGE_OUTPUT` taken from the corresponding config file and `output_path` from *settings.py*.

When the whole pipeline is executed a summary report is created in `output_path/STAGE_NAME/report.html`. For individual stage execution it can be created with `snakemake --report /path/to/report.html`.

__Block input__:
Inputs to specific blocks are handled by the corresponding rule in the *Snakefile* dependent on the mechanics of the respective stage.

__Block outputs__:
All output from blocks (data and figures) is stored hierarchically in `output_path/STAGE_NAME/<block name>/`.

### Global Parameters
[see config](config_template.yaml)

PLOT_TSTART, PLOT_TSTOP, PLOT_CHANNEL, PLOT_FORMAT, USE_LINK_AS_STAGE_OUTPUT, NEO_FORMAT
