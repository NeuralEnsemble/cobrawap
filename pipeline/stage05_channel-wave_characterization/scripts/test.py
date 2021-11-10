"""
Docstring
"""

import argparse
from utils import load_neo, save_plot, none_or_str
from utils import AnalogSignal2ImageSequence


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='Wavefronts', objects="Event")[0]

    """
    compute measure...
    """

    # transform to DataFrame
    df = pd.DataFrame(np.array(wave_id, measure),
                      columns=['wave_id', 'measure_name'],
                      index=channel_ids)
    df['measure_unit'] = [unit]*len(channel_ids)
    df.index.name = 'channel_id'

    df.to_csv(args.output)
