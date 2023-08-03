import os
import neo
import matplotlib.pyplot as plt
import warnings
from snakemake.logging import logger
from pathlib import Path

def load_neo(filename, object='block', lazy=False, *args, **kwargs):
    try:
        filename = Path(filename)
        if filename.suffix == '.nix':
            kwargs.update(mode='ro')

        io = neo.io.get_io(str(filename), *args, **kwargs)

        if lazy and io.support_lazy:
            block = io.read_block(lazy=lazy)
        # elif lazy and isinstance(io, neo.io.nixio.NixIO):
        #     with neo.NixIOFr(filename, *args, **kwargs) as nio:
        #         block = nio.read_block(lazy=lazy)
        else:
            block = io.read_block()

    except Exception as e:
        # io.close()
        raise e
    finally:
        if not lazy and hasattr(io, 'close'):
            io.close()

    if block is None:
        raise IOError(f'{filename} does not exist!')

    if object == 'block':
        return block
    elif object == 'analogsignal':
        return block.segments[0].analogsignals[0]
    else:
        raise IOError(f"{object} not recognized! Choose 'block' or 'analogsignal'.")


def write_neo(filename, block, *args, **kwargs):
    # muting saving imagesequences for now, since they do not yet
    # support array_annotations
    block.segments[0].imagesequences = []
    try:
        # for neo > 0.12.0 filename can't contain '|'
        io = neo.io.get_io(str(filename), *args, **kwargs)
        io.write(block)
    except Exception as e:
        warnings.warn(str(e))
    finally:
        io.close()
    return True


def save_plot(filename, dpi=300):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    try:
        plt.savefig(fname=filename, dpi=dpi, bbox_inches='tight')
    except ValueError as ve:
        warnings.warn(str(ve))
        plt.subplots()
        plt.savefig(fname=filename)
    plt.close()
    return None
