import os
from pathlib import Path
from ruamel.yaml import YAML
from copy import copy
from types import SimpleNamespace
from snakemake.logging import logger
yaml = YAML()
yaml.indent(mapping=4, sequence=4, offset=4)
# Todo: re-evaluate if this is the most elegant solution to hardcode
#       the settings path here
SETTINGS_PATH = Path('~/.cobrawap/config').expanduser()

def get_setting(key: str = None):
    try:
        with open(SETTINGS_PATH, 'r') as f:
            settings = yaml.load(f)
            if not settings:
                settings = {}
    except Exception as e:
        raise FileNotFoundError("The settings file `~/.cobrawap/config` "\
                                "either does not exist "\
                                "or is not in proper JSON format! "\
                                "Try to run `cobrawap init` to set up the "\
                                "settings file.") from e
    if key is None:
        return settings
    elif key not in settings.keys():
        raise ValueError(f"{key} not found in settings file!")
    else:
        return settings[key]


def set_setting(setting: dict, force_overwrite=False) -> None:
    if type(setting) != dict:
        raise TypeError('Function expects a dictionary!')

    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, 'r') as f:
            settings = yaml.load(f)
        if not settings:
            settings = {}
    else:
        logger.warning("Creating a new cobrawap settings file at "\
                      f"`{SETTINGS_PATH.resolve()}`.")
        if not SETTINGS_PATH.parent.exists():
            SETTINGS_PATH.parent.mkdir(parents=True)
        SETTINGS_PATH.touch(exist_ok=True)
        settings = {}

    key_overlap = [k for k in setting.keys() if k in settings.keys()]
    if key_overlap and not force_overwrite:
        overwrite = (input(f"There are already settings for {key_overlap}! "\
                            "Overwrite? [y/N]").lower() == 'y'
                    or False)
        if not overwrite:
            _ = [setting.pop(k, None) for k in key_overlap]

    settings.update(setting)
    with open(SETTINGS_PATH, 'w') as f:
         yaml.dump(settings, f)
    return None


def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def read_stage_output(stage, config_dir, config_name,
                      output_namespace="STAGE_OUTPUT"):
    config_path = Path(config_dir) / stage
    if (config_path / 'configs').is_dir():
        config_path = config_path / 'configs'

    with open(config_path / config_name, 'r') as f:
        config_dict = yaml.load(f)
    if config_dict is None:
        logger.warning('config file '
                      f'{os.path.join(config_dir, stage, config_name)} '
                       'can not be loaded! Skipping reading stage output.')
        return None
    elif output_namespace in config_dict.keys():
        stage_output = config_dict[output_namespace]
        if not Path(stage_output).suffix:
            neo_format = config_dict['NEO_FORMAT']
            stage_output = f'{stage_output}.{neo_format}'
        return stage_output
    else:
        logger.error(f"config file of stage {stage} "
                     f"does not define {output_namespace}!")


def load_config_file(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f)
    if not config_dict:
        logger.error(f"config file is empty: {config_path}")
        raise FileNotFoundError
    return config_dict


def get_parent_config_name(config_name):
    """
    splitting last '_<something>' element
    keeping a trailing '|<anything>'
    """
    name, ext = os.path.splitext(config_name)
    main, *variant = name.split('|')
    variant = variant[0] if variant else ''

    parent = "_".join(main.split('_')[:-1])
    if not parent:
        return False
    elif variant:
        parent = '|'.join([parent, variant])

    return parent + ext


def get_config(config_dir, config_name, get_path_instead=False):
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
    config_dir = Path(config_dir)
    if (config_dir / 'configs').is_dir():
        config_dir = config_dir / 'configs'

    try_config_name = copy(config_name)

    keep_variant = True
    while not config_dict:
        try:
            config_path = config_dir / try_config_name
            config_dict = load_config_file(config_path)
        except FileNotFoundError:
            prev_try_config_name = try_config_name

            parent_config_name = get_parent_config_name(try_config_name)
            if parent_config_name:
                try_config_name = parent_config_name
            else:
                if keep_variant and '|' in try_config_name:
                    name, ext = os.path.splitext(config_name)
                    try_config_name = name.split('|')[0] + ext
                    keep_variant = False
                else:
                    logger.error("No corresponding config file found!")
                    return {}

            logger.info(f"'{prev_try_config_name}' not found, "
                        f"trying '{try_config_name}'.")

    logger.info(f"Using '{try_config_name}.")
    if get_path_instead:
        return config_path
    else:
        return config_dict


def create_temp_configs(stages, configs_dir, config_name, output_dir,
                        temp_name='temp_config.yaml'):
    for i, stage in enumerate(stages):
        config_dict = get_config(os.path.join(configs_dir, stage), config_name)
        new_config_path = os.path.join(output_dir, stage, temp_name)
        with safe_open_w(new_config_path) as f:
            yaml.dump(config_dict, f)
    return None


def update_configfile(config_path, update_dict):
    # Careful! This function overwrites the config file.
    # Comments in the file are lost.
    with open(config_path, 'r') as f:
        try:
            config_dict = yaml.load(f)
        except Exception as e:
            logger.info(e)
            config_dict = None
    if config_dict is None:
        logger.warning(f'config file {config_path} can not be loaded! '
                       'Skipping updating config.')
        return None
    config_dict.update(**update_dict)
    with safe_open_w(config_path) as f:
        yaml.dump(config_dict, f)
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
            logger.warning(f'Could not read stage output for {stage}! '
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


def dict_to_cla(arg_dict):
    for key, value in arg_dict.items():
        if type(value) == list:
            arg_dict[key] = ' '.join(str(v) for v in value)

    cla_str = lambda k,v: f'--{k} "{v}"' if in_quotes(v) else f'--{k} {v}'
    arg_strings = [cla_str(key, value) for key, value in arg_dict.items()]
    return ' '.join(arg_strings)


def in_quotes(object):
    char = [r"\s", "|"]  # when these characters are in an argument str
                         # they need to be put in quotes to be passed as cla
    return any([c in str(object) for c in char])


def params(*args, config=None, **kwargs):
    '''
    Creates a parameter dictionary that is returned as a string as
    command line argument as `--key value` for each key-value pair in the dict.

    args:
        - if args[0] is a dict, it is added to the parameter dict
        - if args are strings and config is a dict, args (and args.upper())
          are interpreted as keys of config dict and the corresponding elements
          are added to the parameter dict
    config:
        dict from which elements (keys given by args) are selected
    kwargs:
        kwargs are added as key-value pairs are to the parameter dict

    Additionally, all wildcards and named outputs of the current rule are added
    as key-value pairs to the parameter dict.
    '''

    param_dict = {}  # use default dict?

    if len(args) and type(args[0]) == dict:
        param_dict = args[0]

    if type(config) == SimpleNamespace:
        config = vars(config)

    if type(config) == dict:
        for arg in args:
            if not type(arg) == str:
                continue
            if arg in config.keys():
                param_dict[arg] = config[arg]
            elif arg.upper() in config.keys():
                param_dict[arg] = config[arg.upper()]
            else:
                logger.info(f'Parameter "{arg}" not found in the config! '
                             'Set to None!')
                param_dict[arg] = None

    param_dict.update(dict(kwargs.items()))

    def add_output_and_wildcards_to_args(wildcards, output):
        for items in [wildcards, output]:
            item_dict = dict(items.items())
            if 'data' in item_dict.keys():
                logger.warning("wildcards or outputs named 'data' are being ignored!")
                del item_dict['data']

            duplicates = [key for key in item_dict.keys() if key in param_dict.keys()]

            for key in duplicates:
                if param_dict[key] != item_dict[key]:
                    logger.warning(f"The keyword {key} is used multiple times "
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
        logger.error(f"Can't find rule '{string}'! Please check the spelling "
                      "and the config file.")
