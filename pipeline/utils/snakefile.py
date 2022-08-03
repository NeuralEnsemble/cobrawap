import os
import yaml
import warnings
from copy import copy
from types import SimpleNamespace
from warnings import warn
from snakemake.logging import logger


def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def read_stage_output(stage, config_dir, config_name,
                      output_namespace="STAGE_OUTPUT"):
    with open(os.path.join(config_dir, stage, config_name), 'r') as f:
        config_dict = yaml.safe_load(f)
    if config_dict is None:
        warnings.warn(f'config file {os.path.join(config_dir, stage, config_name)} '
                       'can not be loaded! Skipping reading stage output.')
        return None
    elif output_namespace in config_dict.keys():
        return config_dict[output_namespace]
    else:
        raise ValueError(f"config file of stage {stage} "
                       + f"does not define {output_namespace}!")


def load_config_file(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_parent_config_name(config_name):
    """
    splitting last '_<something>' element
    keeping a trailing '|<anything>'
    """
    name, ext = os.path.splitext(config_name)
    if '|' in name:
        main, variant = name.split('|')
    else:
        main, variant = name, ''
    parent = "_".join(main.split('_')[:-1])
    if not parent:
        return False
    elif '|' in name:
        return '|'.join([parent,variant]) + ext
    else:
        return parent + ext


def get_config(config_dir, config_name):
    """
    # search order:
    config_some_profile_name|variant.yaml
    config_some_profile|variant.yaml
    config_some|variant.yaml
    config|variant.yaml
    config_some_profile_name.yaml
    config_some_profile.yaml
    config_some.yaml
    config.yaml
    """
    config_dict = {}
    if os.path.isdir(os.path.join(config_dir, 'configs')):
        config_dir = os.path.join(config_dir, 'configs')

    try_config_name = copy(config_name)

    keep_variant = True
    while not config_dict:
        try:
            config_dict = load_config_file(os.path.join(config_dir,
                                                        try_config_name))
        except FileNotFoundError:
            parent_config_name = get_parent_config_name(try_config_name)

            if parent_config_name:
                logger.info(f"{try_config_name} not found, "
                            f"trying {parent_config_name}")
                try_config_name = parent_config_name
            else:
                if keep_variant:
                    name, ext = os.path.splitext(config_name)
                    try_config_name = name.split('|')[0] + ext
                else:
                    raise FileNotFoundError("No corresponding config file found!")

    return config_dict


def create_temp_configs(stages, configs_dir, config_name, output_dir,
                        temp_name='temp_config.yaml'):
    for i, stage in enumerate(stages):
        config_dict = get_config(os.path.join(configs_dir, stage), config_name)
        new_config_path = os.path.join(output_dir, stage, temp_name)
        with safe_open_w(new_config_path) as f:
            f.write(yaml.dump(config_dict, default_flow_style=False))
    return None


def update_configfile(config_path, update_dict):
    # Careful! This function overwrites the config file.
    # Comments in the file are lost.
    with open(config_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
        except Exception as e:
            logger.info(e)
            config_dict = None
    if config_dict is None:
        warnings.warn(f'config file {config_path} can not be loaded! '
                       'Skipping updating config.')
        return None
    config_dict.update(**update_dict)
    with safe_open_w(config_path) as f:
        f.write(yaml.dump(config_dict, default_flow_style=False))
    return None


def set_stage_inputs(stages, output_dir, config_file='temp_config.yaml',
                     input_namespace="STAGE_INPUT"):
    update_dict = {}
    update_configfile(config_path=os.path.join(output_dir, stages[0], config_file),
                              update_dict={input_namespace:None})

    for i, stage in enumerate(stages[:-1]):
        output_name = read_stage_output(stage,
                                        config_dir=output_dir,
                                        config_name=config_file)
        if output_name is None:
            warnings.warn(f'Could not read stage output for {stage}! '
                           'Skipping setting input for subsequent stage.')
        else:
            update_dict[input_namespace] = os.path.join(output_dir, stage, output_name)
            update_configfile(config_path=os.path.join(output_dir, stages[i+1], config_file),
                              update_dict=update_dict)
    return None


def set_global_configs(stages, output_dir, config_dict,
                       config_file='temp_config.yaml'):
    for stage in stages:
        update_configfile(config_path=os.path.join(output_dir, stage, config_file),
                          update_dict=config_dict)
    return None


def get_param(config, param_name):
    if param_name in config:
        return config[param_name]
    else:
        return None


def dict_to_cla(arg_dict):
    for key, value in arg_dict.items():
        if type(value) == list:
            arg_dict[key] = ' '.join(str(v) for v in value)

    cla_str = lambda k,v: f'--{k} "{v}"' if is_path(v) else f'--{k} {v}'
    arg_strings = [cla_str(key, value) for key, value in arg_dict.items()]
    return ' '.join(arg_strings)


def is_path(object):
    return '/' in str(object) or '\\' in str(object)


def params(*args, config=None, **kwargs):
    '''
    creates parameter dictionary to pass to script as `--key value`
    - if args[0] is a dict, use it
    - args are interpreted as keys for the config dict (non case-sensitive)
        - if config if given, else args are ignored
    - kwargs are interpreted as key-value pairs
    '''
    param_dict = {}  # use default dict?

    if len(args) and type(args[0]) == dict:
        param_dict = args[0]

    if type(config) == SimpleNamespace:
        config = vars(config)

    if config is not None:
        for arg in args:
            if not type(arg) == str:
                continue
            if arg in config.keys():
                param_dict[arg] = config[arg]
            elif arg.upper() in config.keys():
                param_dict[arg] = config[arg.upper()]
            else:
                print(f'Parameter {arg} not found in the config! Set to None!')
                param_dict[arg] = None

    param_dict.update(dict(kwargs.items()))

    def add_output_and_wildcards_to_args(wildcards, output):
        for items in [wildcards, output]:
            item_dict = dict(items.items())
            if 'data' in item_dict.keys():
                warn("wildcards or outputs name 'data' are being ignored!")
                del item_dict['data']

            duplicates = [key for key in item_dict.keys() if key in param_dict.keys()]
            
            for key in duplicates:
                if param_dict[key] != item_dict[key]:
                    warn("The keyword {key} is used multiple times "
                         "in the rule's params, wildcards, or output! \n"
                        f"{key}: '{param_dict[key]}' is ignored "
                        f"in favor of '{item_dict[key]}'")

            param_dict.update(item_dict)
        return dict_to_cla(param_dict) 

    return add_output_and_wildcards_to_args


def locate_str_in_list(str_list, string):
    if string in str_list:
        return [i for i, el in enumerate(str_list) if el == string][0]
    else:
        raise ValueError(f"Can't find rule '{string}'! Please check the spelling"
                          "and the config file.")
