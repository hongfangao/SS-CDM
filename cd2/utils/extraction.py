import re
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from cd2.dataloaders.datamodules import DataModule
from cd2.models.score_models import ScoreModule, CD2

def get_training_params(datamodule: DataModule, trainer: pl.Trainer) -> dict[str, Any]:
    params = datamodule.dataset_parameters
    params["num_training_steps"] *= trainer.max_epochs
    params["num_training_steps"] /= trainer.accumulate_grad_batches
    assert(isinstance(params,dict))
    return params

def flatten_config(cfg: DictConfig|dict) -> dict[str, Any]:
    cfg_dict = (
        OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    )
    assert(isinstance(cfg_dict,dict))

    cfg_flat: dict[str, Any] = {}
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            if "_target_" in v:
                cfg_flat[k] = v["_target_"]
            cfg_flat.update(**flatten_config(v))
        elif isinstance(v, list):
            v_ls = []
            for v_i in v:
                if isinstance(v_i, dict):
                    v_ls.append(v_i["_target_"])
                cfg_flat.update(**flatten_config(v_i))
            cfg_flat[k] = v_ls
        elif k not in {"_target_","_partial_"}:
            cfg_flat[k] = v
    
    return cfg_flat

def get_model_type(cfg: DictConfig|dict) -> ScoreModule | CD2:
    model_class = cfg["score_model"]["_target_"]
    match model_class:
        case "ScoreModule":
            return ScoreModule
        case "CD2":
            return CD2
        case _:
            raise NotImplementedError(f"Model class {model_class} not implemented")
        
def get_best_ckpt(checkpoint_path: Path) -> Path:
    pattern = r"(.+?)epoch=(\d+)-val_loss=(\d+\.\d+).ckpt"
    best_loss = float("inf")
    for checkpoint in checkpoint_path.glob("*.ckpt"):
        match = re.match(pattern, str(checkpoint))
        if match is not None:
            loss = float(match.group(3))
            if loss < best_loss:
                best_loss = loss
                best_checkpoint_path = checkpoint
    return best_checkpoint_path


def dict_to_str(dict: DictConfig | dict[str, Any]) -> str:
    """Convert a dict to a string with breaklines.

    Args:
        dict (DictConfig | dict[str, Any]): Dictionary to convert.

    Returns:
        str: String describing the dictionary's content line by line.
    """

    if isinstance(dict, DictConfig):
        dict = flatten_config(dict)

    dict_str = ""
    max_len = max([len(k) for k in dict])
    for k, v in dict.items():
        # In case of long lists, just print the first 3 elements
        if isinstance(v, list):
            v = v[:3] + ["..."] if len(v) > 3 else v
        dict_str += f"\t {k: <{max_len + 5}} : \t  {v} \t \n"
    return dict_str
