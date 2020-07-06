import argparse
import numpy as np
import neo
from utils import load_neo, write_neo, AnalogSignal2ImageSequence


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--trigger_data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--node_data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")

    args = CLI.parse_args()
    trigger_block = load_neo(args.trigger_data)
    node_block = load_neo(args.node_data)

    block = AnalogSignal2ImageSequence(node_block)

    wavefront_evt = [evt for evt in trigger_block.segments[0].events
                     if evt.name == "Wavefronts"]
    if wavefront_evt:
        wavefront_evt = wavefront_evt[0]
    else:
        raise ValueError("Input does not contain an event with name " \
                       + "'Wavefronts'!")

    block.segments[0].events.append(wavefront_evt)

    write_neo(args.output, block)
