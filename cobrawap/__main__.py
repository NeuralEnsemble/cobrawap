#!/usr/bin/env python
# encoding: utf8
'''
Collaborative Brain Wave Analysis Pipeline (Cobrawap)
'''

import sys
import logging
import argparse
import subprocess
import shutil
import re
from pprint import pformat
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from cmd_utils import get_setting, set_setting, get_initial_available_stages
from cmd_utils import is_profile_name_valid, create_new_configfile 
from cmd_utils import input_profile, get_profile, setup_entry_stage, working_directory
log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# requires hardcoded location where to lookup all defined paths
try:
    STAGES = get_setting('stages')
except Exception as e:
    log.debug(e)
    try:
        STAGES = get_initial_available_stages()
    except Exception as e:
        log.debug(e)
        STAGES = {}

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
CLI_init.add_argument("--output_path", type=Path, default=None,
                      help="directory where the analysis output is stored "
                           "[default: '~/cobrawap_output/']")
CLI_init.add_argument("--config_path", type=Path, default=None,
                      help="directory where the analysis config files are "
                           "stored [default: '~/cobrawap_config/']")

# Configuration
CLI_create = subparsers.add_parser('create', 
                        help='create configuration for a new dataset')
CLI_create.set_defaults(command='create')
CLI_create.add_argument("--data_path", type=Path, nargs='?', default=None,
                        help="full path to the dataset to be analyzed "
                             "(or where it will be stored)")
CLI_create.add_argument("--loading_script_name", nargs='?', type=Path, default=None,
                        help="name of the data specific loading script "
                             "(in <config_path>/stage01_data_entry/scripts/)")
CLI_create.add_argument("--profile", type=str, nargs='?', default=None,
                        help="profile name of this dataset/application "
                             "(see profile name conventions in documentation)")
CLI_create.add_argument("--parent_profile", type=str, nargs='?', default=None, 
                        help="optional parent profile name "
                             "(see profile name conventions in documentation)")

# Additional configurations
CLI_profile = subparsers.add_parser('add_profile', 
                         help='create a new configuration for an existing dataset')
CLI_profile.set_defaults(command='add_profile')
CLI_profile.add_argument("--profile", type=str, nargs='?', default=None,
                        help="profile name of this dataset/application "
                             "(see profile name conventions in documentation)")
CLI_profile.add_argument("--stages", type=str, nargs='*',
                         choices=list(STAGES.keys()), default=None,
                         help="selection of pipeline stages to configure")
CLI_profile.add_argument("--parent_profile", type=str, nargs='?', default=None,
                         help="optional parent profile name from which to "
                              "initialize the new config "
                              "[default: basic template]")

# Run
CLI_run = subparsers.add_parser('run', 
                            help='run the analysis pipeline on the selected '
                                 'input and with the specified configurations')
CLI_run.set_defaults(command='run')
CLI_run.add_argument("--profile", type=str, nargs='?', default=None,
                     help="profile name of the dataset to be analyzed")


def main():
    'Start main CLI entry point.'
    args, unknown = CLI.parse_known_args()
    # log.info("this is a regular print statement")
    # log.debug("this is a verbose print statement")

    if args.verbose:
        log.setLevel(logging.DEBUG)
    log.debug(pformat(args))

    if args.command == 'init':
        log.info("initializing Cobrawap")
        initialize(**vars(args))
        
    elif args.command == 'create':
        log.info("creating a set of config files")
        create(**vars(args))

    elif args.command == 'add_profile':
        log.info("creating a new config files")
        add_profile(**vars(args))

    elif args.command == 'run':
        log.info("executing Cobrawap")
        run(**vars(args), extra_args=unknown)

    else:
        log.info(f"{args.command} not known!")

    return None


def initialize(output_path=None, config_path=None, **kwargs):
    # set output_path
    if output_path is None:
        output_path = Path(input("Output directory "\
                                 "[default: ~/cobrawap_output]:")
                           or Path('~') / 'cobrawap_output').expanduser()
    output_path.mkdir(exist_ok=True)
    if not output_path.is_dir():
        raise ValueError(f"{output_path} is not a valid directory!")

    set_setting(dict(output_path=str(output_path.resolve())))

    # set config_path
    if config_path is None:
        config_path = Path(input("Config directory "\
                                 "[default: ~/cobrawap_configs]: ") \
                      or Path('~') / 'cobrawap_configs').expanduser()
    config_path.mkdir(parents=True, exist_ok=True)
    if not config_path.is_dir():
        raise ValueError(f"{config_path} is not a valid directory!")
            
    set_setting(dict(config_path=str(config_path.resolve())))

    # set pipeline path
    pipeline_path = Path(__file__).parents[1] / 'pipeline'
    set_setting(dict(pipeline_path=str(pipeline_path.resolve())))

    # set available stages
    set_setting(dict(stages=get_initial_available_stages()))
    stages = get_setting('stages')

    # populate config_path with template config files
    for stage_number, stage in stages.items():
        stage_config_path = config_path / stage / 'configs'
        stage_config_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(pipeline_path / stage / 'configs' / 'config_template.yaml',
                    stage_config_path / 'config.yaml')
    
    pipeline_config_path = config_path / 'configs'
    pipeline_config_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(pipeline_path / 'configs' / 'config_template.yaml',
                pipeline_config_path / 'config.yaml')
    
    stage01_script_path = config_path / stages['1'] / 'scripts'
    stage01_script_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(pipeline_path / stages['1'] / 'scripts' / 'enter_data_template.py',
                stage01_script_path / 'enter_data_template.py')

    return None


def create(profile=None, parent_profile=None, data_path=None, 
           loading_script_name=None, **kwargs):
    profile, parent_profile = get_profile(profile=profile,  
                                          parent_profile=parent_profile)
    base_name = parent_profile if parent_profile else profile

    for stage_number, stage in get_setting('stages').items():
        config_name = profile if '1' in str(stage_number) else base_name
        create_new_configfile(stage=stage, 
                              profile=config_name,
                              parent=parent_profile)
        
    setup_entry_stage(profile=profile, parent_profile=parent_profile,
                  data_path=data_path, loading_script_name=loading_script_name)
    return None


def add_profile(profile=None, parent_profile=None, data_path=None, 
                stages=None, loading_script_name=None, **kwargs):
    profile, parent_profile = get_profile(profile=profile, 
                                          parent_profile=parent_profile)
    # get stage selection
    stages = ''
    while not stages:
        stages = input("To which stages should this profile be applied? "
                      f"{list(get_setting('stages').keys())}:")
        try:
            stages = stages.replace("'","")
            stages = re.split(',|\s', stages)
            stages = [stage for stage in stages if stage]
        except Exception as e:
            log.info(e)
            stages = ''

    for stage_number in stages:
        create_new_configfile(stage_number=stage_number, 
                              profile=profile,
                              parent=parent_profile)
        
    if any('1' in stage for stage in stages):
        setup_entry_stage(profile=profile, parent_profile=parent_profile,
                          data_path=data_path, 
                          loading_script_name=loading_script_name)
    return None


def run(profile=None, extra_args=None, **kwargs):
    # select profile
    if profile is not is_profile_name_valid(profile) and profile is not None:
        log.info(f"profile name {profile} is not valid!")
        profile = None
    if profile is None:
        profile = input_profile()

    # set runtime config
    pass 

    # execute snakemake
    with working_directory(Path(get_setting('pipeline_path'))):
        subprocess.run(['snakemake','-c1','--config',f'PROFILE="{profile}"']
                       + extra_args)

    return None


if __name__ == '__main__':
    main()
