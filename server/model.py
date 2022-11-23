import torch

from omegaconf.dictconfig import DictConfig
from typing import List, Callable

def load_model_from_config(config: DictConfig, ckpt: str):
    """This function will take in a model config and model checkpoint file and import all
    the necessary libraries for that model and load the model from the torch ckpt file.

    Args:
        config (omegaconfig.dictconfig.DictConfig): Omegaconfig for the current model
        ckpt (str): Path to the model torch ckpt file

    Raises:
        KeyError: If "target" value of configuration is not valid

    Returns:
        _type_: _description_
    """
    torch_model = torch.load(ckpt, map_location="cpu")
    state_dict = torch_model["state_dict"]
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    model, cls = config["target"].rsplit(".", 1)

def txt2img(model: Callable, prompts: List[str]):
    return