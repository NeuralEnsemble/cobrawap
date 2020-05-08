## in rule
* for using functions from `utils.py` add `{ADD_UTILS}` as first line of the shell command
* Each rule should have an additional plotting output.
    * If applicable, make use of config parameters `PLOT_TSTART`, `PLOT_TSTOP`, `PLOT_CHANNEL`
    * When the plot is 'general' use a separate dedicated plotting script,
    otherwise add an optional plotting output to the script.
* state all parameters and global variables you use explicitly as `params`

example rule:
```
rule enter_data:
    input:
        data = input_file,
        script = os.path.join('scripts/', CURATION_SCRIPT),
        plot_script = 'scripts/plot_traces.py'
    output:
        data = os.path.join(OUTPUT_DIR, '{data_name}.'+NEO_FORMAT),
        img = report(os.path.join(OUTPUT_DIR, 'trace_{data_name}.'+PLOT_FORMAT))
    params:
        spatial_scale = SPATIAL_SCALE,
        sampling_rate = SAMPLING_RATE,
        t_start = T_START,
        t_stop = T_STOP,
        annotations = ANNOTATIONS,
        array_annotations = ARRAY_ANNOTATIONS,
        kwargs = KWARGS,
        plot_tstart = PLOT_TSTART,
        plot_tstop = PLOT_TSTOP,
        plot_channels = PLOT_CHANNELS
    shell:
        """
        {ADD_UTILS}
        python {input.script} --data "{input.data}" \
                              --output "{output.data}" \
                              --sampling_rate {params.sampling_rate} \
                              --spatial_scale {params.spatial_scale} \
                              --t_start {params.t_start} \
                              --t_stop {params.t_stop} \
                              --data_name {wildcards.data_name} \
                              --annotations {params.annotations} \
                              --array_annotations {params.array_annotations} \
                              --kwargs {params.kwargs}
        python {input.plot_script} --data "{output.data}" \
                                   --output "{output.img}" \
                                   --t_start {params.plot_tstart} \
                                   --t_stop {params.plot_tstop} \
                                   --channels {params.plot_channels} \
        """
```


## in script
* the `if __name__ == '__main__':` section should only contain
    1. argument handling (using `argparse`)
    2. handling of exceptions, warnings, errors
    3. functions calls
* make use of the utility functions (`import utils`), e.g. load_neo, save_plot, or the custom argparse types `none_or_int` etc.
* make the script the most usable also as a standalone
    * write docstring
    * argparse arguments should have `default` values (or be set to `required=True`)
    * handle multiple input argument options, e.g. only create plots when `args.output_img is not None`
    * handle exceptions and print warnings or raise errors
* Write a docstring using *???* convention

example script:
```
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import quantities as pq
import random
from utils import load_neo, save_plot, time_slice, parse_plot_channels,\
                  none_or_int


def plot_traces(asig, channels):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    offset = np.max(np.abs(asig.as_array()[:,channels]))

    for i, signal in enumerate(asig.as_array()[:,channels].T):
        ax.plot(asig.times, signal + i*offset)

    annotations = [f'{k}: {v}' for k,v in asig.annotations.items()
                               if k not in ['nix_name', 'neo_name']]
    array_annotations = [f'{k}: {v[channels]}'
                        for k,v in asig.array_annotations.items()]

    ax.text(ax.get_xlim()[1]*1.05, ax.get_ylim()[0],
            f'ANNOTATIONS FOR CHANNEL(s) {channels} \n'\
            + '\n ANNOTATIONS:\n' + '\n'.join(annotations) \
            + '\n\n ARRAY ANNOTATIONS:\n' + '\n'.join(array_annotations))

    ax.set_xlabel(f'time [{asig.times.units.dimensionality.string}]')
    ax.set_ylabel(f'channels [in {asig.units.dimensionality.string}]')
    ax.set_yticks([i*offset for i in range(len(channels))])
    ax.set_yticklabels(channels)
    return fig


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output figure")
    CLI.add_argument("--t_start", nargs='?', type=float, default=0,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=float, default=10,
                     help="stop time in seconds")
    CLI.add_argument("--channels", nargs='+', type=none_or_int, default=0,
                     help="list of channels to plot")
    args = CLI.parse_args()

    asig = load_neo(args.data, 'analogsignal', lazy=True)

    channels = parse_plot_channels(args.channels, args.data)

    asig = time_slice(asig, t_start=args.t_start, t_stop=args.t_stop,
                      lazy=True, channel_indexes=channels)

    fig = plot_traces(asig, channels)

    save_plot(args.output)
```
