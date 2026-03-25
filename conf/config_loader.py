# In conf/config_loader.py

from omegaconf import OmegaConf
from .config_schema import Config

def load_app_config(config_path: str = 'conf') -> Config:
    """
    A simple, direct function to load and merge all configurations.
    """
    # Load the base and command-line configs first
    base_conf = OmegaConf.load(f"{config_path}/base.yaml")
    cli_conf = OmegaConf.from_cli()
    
    # Temporarily merge them to find out which algorithm to use
    temp_conf = OmegaConf.merge(base_conf, cli_conf)
    algo_name = temp_conf.algorithm.name

    # Build a list of YAML files to load in the correct order
    #    (from least specific to most specific)
    yaml_files = [
        f"{config_path}/base.yaml",
        f"{config_path}/algorithm/{algo_name}.yaml"
    ]

    # If using SSFL, we need to also load its specific mode file
    if algo_name == 'ssfl':
        # To find the final mode, we merge the YAMLs found so far with the CLI
        mode_check_conf = OmegaConf.merge(
            *[OmegaConf.load(f) for f in yaml_files], cli_conf
        )
        ssfl_mode = mode_check_conf.algorithm.params.mode
        yaml_files.append(f"{config_path}/algorithm/mode/{ssfl_mode}.yaml")

    # Load all determined YAML files
    configs_from_yaml = [OmegaConf.load(f) for f in yaml_files]

    # Merge everything: YAML files first, then CLI for final overrides
    final_conf = OmegaConf.merge(*configs_from_yaml, cli_conf)

    # Apply the strict schema at the very end
    schema = OmegaConf.structured(Config)
    return OmegaConf.merge(schema, final_conf)