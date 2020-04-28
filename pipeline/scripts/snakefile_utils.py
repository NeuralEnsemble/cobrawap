import os
import yaml

def read_stage_output(stage, working_dir, output_namespace="STAGE_OUTPUT"):
    with open(os.path.join(working_dir, stage, 'config.yaml'), 'r') as f:
        config_dict = yaml.safe_load(f)
    if output_namespace in config_dict.keys():
        return config_dict[output_namespace]
    else:
        raise ValueError(f"config file of stage {stage} "
                       + f"does not define {output_namespace}!")

def create_temp_configs(stages, working_dir, file_name='temp_config.yaml'):
    for i, stage in enumerate(stages):
        config_path = os.path.join(working_dir, stage, 'config.yaml')
        new_config_path = os.path.join(working_dir, stage, file_name)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        with open(new_config_path, 'w') as f:
            f.write(yaml.dump(config_dict, default_flow_style=False))
    return None

def update_configfile(config_path, update_dict):
    # Careful! This function overwrites the config file.
    # Comments in the file are lost.
    with open(config_path, 'r+') as f:
        config_dict = yaml.safe_load(f)
        config_dict.update(**update_dict)
        f.write(yaml.dump(config_dict, default_flow_style=False))
    return None

def set_stage_inputs(stages, working_dir, output_path, config_file='temp_config.yaml',
                     intput_namespace="STAGE_INPUT"):
    update_dict = {}
    for i, stage in enumerate(stages[:-1]):
        output_name = read_stage_output(stage, working_dir=working_dir)
        update_dict[intput_namespace] = os.path.join(output_path, stage, output_name)
        update_configfile(config_path=os.path.join(working_dir, stages[i+1], config_file),
                          update_dict=update_dict)
    return None

def set_global_configs(stages, working_dir, config_dict, config_file='temp_config.yaml'):
    for stage in stages:
        update_configfile(config_path=os.path.join(working_dir, stage, config_file),
                          update_dict=config_dict)
    return None
