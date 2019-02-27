import neo
import argparse
import os
import quantities as pq


def load(data):
    obj = neo.io.spike2io.Spike2IO(data)
    segment = obj.read_segment()
    for (i, asig) in enumerate(segment.analogsignals):
        segment.analogsignals[i] = segment.analogsignals[i] * pq.mV
    return segment


def load_segment(filename):
    # to be called from other scripts to load the nix file
    f = neo.io.nixio.NixIO(filename)
    block = f.read_block()
    return block.segments[0]


def enrich(segment, metadata):
    if os.path.exists(metadata):
        from metadata import electrode_location, electrode_color
        for (i, asig) in enumerate(segment.analogsignals):
            for element in electrode_location:
                if asig.annotations['physical_channel_index'] + 1 \
                        in electrode_location[element]:
                    segment.analogsignals[i].annotations['cortical_location'] = element
                    segment.analogsignals[i].annotations['electrode_color'] = electrode_color[element]
    else:
        pass
    return segment


def save_segment(segment, location):
    data_dir = os.path.dirname(location)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    block = neo.core.Block()
    block.segments.append(segment)
    nix = neo.io.nixio.NixIO(location)
    nix.write_block(block)
    return None


if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs=1, type=str)
    CLI.add_argument("--data",      nargs=1, type=str)
    CLI.add_argument("--metadata",  nargs=1, type=str, default='')
    args = CLI.parse_args()

    segment = load(data=args.data[0])

    segment = enrich(segment, metadata=args.metadata[0])

    save_segment(segment, location=args.output[0])
