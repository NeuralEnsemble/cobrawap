import contextlib
import inspect
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from pprint import pformat
from string import ascii_lowercase

sys.path.append(str(Path(inspect.getfile(lambda: None))))
from pipeline.utils.snakefile import (
    get_config,
    get_setting,
    load_config_file,
    locate_str_in_list,
    read_stage_output,
    set_setting,
    update_configfile,
)

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def create_new_configfile(profile, stage=None, stage_number=None, parent=None):
    if stage is None:
        if stage_number is None:
            raise KeyError("You must either specify `stage` or `stage_number`!")
        else:

            stage = get_setting("stages")[stage_number]

    config_path = Path(get_setting("config_path"))

    parent = f"_{parent}" if parent else ""
    config_dir = config_path / stage / "configs"
    template_config_path = config_dir / f"config{parent}.yaml"
    new_config_path = config_dir / f"config_{profile}.yaml"

    if new_config_path.exists():
        log.debug(f"config file `{new_config_path}` already exists. Skip.")
        return None

    if not template_config_path.exists():
        log.debug(f"parent config file `{template_config_path}` doesn't exist.")
        potential_parents = [
            f.name
            for f in config_dir.iterdir()
            if f.name.startswith(f"config{parent}")
        ]
        if potential_parents:
            log.debug(f"Using `{potential_parents[0]}` instead.")
            template_config_path = config_dir / potential_parents[0]
        else:
            raise FileNotFoundError(
                "No parent config file fitting the name "
                f"`{parent.strip('_')}` found!"
            )

    shutil.copy(template_config_path, new_config_path)
    update_configfile(new_config_path, {"PROFILE": profile})
    return None


def input_profile(profile=None):
    if not is_profile_name_valid(profile) and profile is not None:
        log.info(f"profile name {profile} is not valid!")
        profile = None
    if profile is None:
        while not is_profile_name_valid(profile):
            profile = input("profile name: ")
            if not is_profile_name_valid(profile):
                log.info(
                    f"{profile} is not a valid profile name. "
                    "Please use only letters, "
                    "'_' (for sub-profile structuring), "
                    "or '|' (for variant separation)."
                )
    return profile


def input_stage(stage=None):
    stages = get_setting("stages")

    while stage not in stages.keys() and stage not in stages.values():
        stage = input(
            "Which stage should be executed?\n    "
            + "\n    ".join(f"{k} {v}" for k, v in stages.items())
            + "\nSelect the stage index or the stage name: "
        )
    if stage in stages.keys():
        stage = stages[stage]
    return stage


def input_block(stage=None, block=None):
    if stage is None:
        raise ValueError("Stage must be specified!")

    block_dir = Path(get_setting("pipeline_path")) / stage / "scripts"
    available_blocks = get_available_blocks(block_dir)

    while block not in available_blocks:

        if block:
            print(f"Block '{block}' is not found in {stage}!")
        else:
            print("Which block should be executed?")

        print(f"Available blocks are: {', '.join(available_blocks)}")
        block = input("Select block: ")

    return block


def print_settings(*args, **kwargs):
    print(pformat(get_setting()))
    return None


def get_available_blocks(block_dir):
    available_scripts = [
        script for script in block_dir.iterdir() if os.path.isfile(script)
    ]
    available_blocks = [
        s.stem
        for s in available_scripts
        if not (s.stem).startswith("_")
        and s.suffix == ".py"
        and "template" not in s.stem
    ]
    return available_blocks


def is_profile_name_valid(profile: str) -> bool:
    if isinstance(profile, str):
        profile = profile.strip("'\"")
        pattern = re.compile(r"[\w\d\|]+")
        return bool(pattern.fullmatch(profile))
    else:
        return False


def get_profile(profile=None, parent_profile=None):
    # set initial profile name
    profile = input_profile(profile=profile)

    # set full profile name
    if parent_profile is None:
        parent_profile = input("Specify the parent profile name [optional]:")

    if (
        is_profile_name_valid(parent_profile)
        and "|" not in parent_profile
        and not profile.startswith(parent_profile)
    ):
        profile = f"{parent_profile}_{profile}"
    elif parent_profile:
        log.info(f"{parent_profile} is not a valid profile parent name. Ignore.")
        parent_profile = None
    return profile, parent_profile


def setup_entry_stage(
    profile, parent_profile=None, data_path=None, loading_script_name=None
):
    """
    Populate an existing config file for stage 01 with a data path and create
    a corresponding loading script from a template.
    """
    config_path = Path(get_setting("config_path"))

    stage01_update_dict = {}

    # set data path
    if data_path is None:
        data_path = (
            input(
                "Path to dataset [optional, "
                "can be set later in the stage01 config file]:"
            )
            or None
        )
    if data_path is not None:
        stage01_update_dict.update({"DATA_SETS": {profile: data_path}})

    # set loading script path
    if loading_script_name is None:
        base_profile = parent_profile if parent_profile else profile

        loading_script_name = (
            input(
                "Loading script name (to be created) "
                "in <config_path>/stage01_data_entry/scripts/ "
                "[default: enter_<profile>.py]:"
            )
            or f"enter_{base_profile}.py"
        )
        if not Path(loading_script_name).suffix == ".py":
            loading_script_name = f"{str(Path(loading_script_name).stem)}.py"
        stage01_update_dict.update({"CURATION_SCRIPT": loading_script_name})

    stages = get_setting("stages")
    loading_script_path = (
        config_path / stages["1"] / "scripts" / f"{loading_script_name}"
    )
    if loading_script_path.exists():
        log.info(f"`{loading_script_name}` already exists. Skip.")
    else:
        shutil.copy(
            config_path / stages["1"] / "scripts" / "enter_data_template.py",
            loading_script_path,
        )

    # update stage 01 config
    stage01_config_path = (
        config_path / stages["1"] / "configs" / f"config_{profile}.yaml"
    )
    update_configfile(stage01_config_path, stage01_update_dict)
    return None


def get_initial_available_stages():
    pipeline_path = Path(get_setting("pipeline_path"))
    stages = [
        x.name
        for x in pipeline_path.iterdir()
        if x.is_dir() and str(x.name).startswith("stage") and "template" not in x.name
    ]
    stages.sort()
    stage_numbers = [int(re.findall(r"\d+", stage)[0]) for stage in stages]
    stage_keys = []
    for i, v in enumerate(stage_numbers):
        totalcount = stage_numbers.count(v)
        count = stage_numbers[:i].count(v)
        stage_keys.append(
            str(v) + ascii_lowercase[count] if totalcount > 1 and count > 0 else str(v)
        )
    return {k: v for k, v in zip(stage_keys, stages)}


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
    return None
