import argparse
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from utils import load_neo, write_neo, none_or_str, save_plot, \
                  ImageSequence2AnalogSignal, AnalogSignal2ImageSequence


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")


    args = CLI.parse_args()
    block = load_neo(args.data)

    block = AnalogSignal2ImageSequence(block)

    imgseq = [im for im in block.segments[0].imagesequences
                        if im.name == "Optical Flow"]
    if imgseq:
        imgseq = imgseq[0]
    else:
        raise ValueError("Input does not contain a signal with name " \
                       + "'Optical Flow'!")

    frames = imgseq.as_array()

    # block = ImageSequence2AnalogSignal(block)
    write_neo(args.output, block)
