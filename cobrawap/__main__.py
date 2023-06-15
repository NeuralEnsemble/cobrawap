#!/usr/bin/env python
# encoding: utf8
'''
Collaborative Brain Wave Analysis Pipeline (Cobrawap)
'''

import logging
from pprint import pformat
import argparse
from pathlib import Path

CLI = argparse.ArgumentParser(prog='cobrawap')

# Utility arguments
CLI.add_argument("-v", "--verbose", action='store_true',
                help="print additional logging information")
CLI.add_argument("-V", "--version", action='version')
CLI.set_defaults(command=None)

# Initialization
subparsers = CLI.add_subparsers(help='')
CLI_init = subparsers.add_parser('init', 
                help='initialize the cobrawap directories (required only once)')
CLI_init.set_defaults(command='init')
CLI_init.add_argument("--settings_path", type=Path, default=None,
                      help="path where cobrawap path information are stored "
                           "[default: '~/.cobrawap/config']")
CLI_init.add_argument("--output_path", type=Path, default=None,
                      help="directory where the analysis output is stored "
                           "[default: '~/cobrawap_output/']")
CLI_init.add_argument("--config_path", type=Path, default=None,
                      help="directory where the analysis config files are "
                           "stored [default: '~/cobrawap_config/']")

# Configuration
CLI_create = subparsers.add_parser('create', 
                help='create configuration for a new dataset/profile')
CLI_create.set_defaults(command='create')
CLI_create.add_argument("--data_path", type=Path, nargs='?', default=None,
                        help="full path to the dataset to be analyzed "
                             "(or where it will be stored)")
CLI_create.add_argument("--loading_script_path", nargs='?', type=Path, default=None,
                        help="full path to the data specific loading script "
                             "(or where to create a template for it)")
CLI_create.add_argument("--stages", type=str, nargs='*', 
                        choices=['1','2','3','4','5a','5b'], default=None,
                        help="selection of pipeline stages to be executed")
CLI_create.add_argument("--profile", type=str, nargs='?', default=None,
                        help="profile name of this dataset/application "
                             "(see profile name conventions in documentation)")
CLI_create.add_argument("--sub-profile-of", type=str, nargs='?', default=None, 
                        help="optional parent profile name "
                             "(see profile name conventions in documentation)")

# Additional configurations
CLI_profile = subparsers.add_parser('add-profile', 
                         help='create configuration for a new dataset '
                              'or application')
CLI_profile.set_defaults(command='add-profile')
CLI_profile.add_argument("--for-stages", type=str, nargs='*',
                         choices=['1','2','3','4','5a','5b'], default=None,
                         help="selection of pipeline stages to be executed")
CLI_profile.add_argument("--copy-from-profile", type=str, nargs='?', default=None,
                         help="profile name from which to initialize the new " 
                              "config [default: basic template]")

# Run
CLI_run = subparsers.add_parser('run', 
                help='run the analysis pipeline on the selected input and with'
                     'the specified configurations')
CLI_run.set_defaults(command='run')
CLI_run.add_argument("--profile", type=str, nargs='?', default=None,
                     help="profile name of the dataset to be analyzed")


log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def main():
    'Start main CLI entry point.'
    args, unknown = CLI.parse_known_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    log.debug(pformat(args))

    if args.command == 'init':
        log.info("initializing Cobrawap")
        initialize(**vars(args))
        
    if args.command == 'create':
        log.info("creating a set of config files")
        create()

    if args.command == 'add-profile':
        log.info("creating a new config files")
        create()

    if args.command == 'run':
        log.info("executing Cobrawap")
        run(profile=args.profile)

    # log.info("this is a regular print statement")
    # log.debug("this is a verbose print statement")

    return None


def initialize(output_path=None, config_path=None, **kwargs):
    if output_path is None:
        output_path = Path(input("Output directory: "))

    if config_path is None:
        config_path = Path(input("Config directory [install dir]: ")) \
                      or Path(__file__).parents(1) / 'pipeline' # Todo: fix!

    breakpoint()
    # set user config folder path

    # write settings into ~/.cobrawap/config
    return None


def create():

    return None


def run():
    # select profile

    # set runtime config

    # add optional snakemake parameters

    # execute the pipeline

    return None


if __name__ == '__main__':
    main()
