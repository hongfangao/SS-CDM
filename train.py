import logging
import os
from functools import partial
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cd2.dataloaders.datamodules import DataModule
from cd2.models.score_models import ScoreModule, CD2
from cd2.utils.callbacks import SamplingCallback
from cd2.utils.extraction import dict_to_str, get_training_params
from cd2.utils.init_wandb import maybe_initialize_wandb

class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        
        logging.info(
            f"Start traning with config:\n{dict_to_str(cfg)}"
        )

        run_id = maybe_initialize_wandb(cfg)

        self.model: ScoreModule | CD2 = instantiate(cfg.model)
        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.datamodule: DataModule = instantiate(cfg.datamodule)

        save_dir = Path.cwd() / "lightning_logs" / run_id
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving logs to {save_dir}.")
        OmegaConf.save(config=cfg, f=save_dir / "train_config.yaml")

        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        if isinstance(self.model, partial):
            training_params = get_training_params(self.datamodule, self.trainer)
            self.model = self.model(**training_params)
        
        for callback in self.trainer.callbacks:
            if isinstance(callback, SamplingCallback):
                callback.setup_datamodule(datamodule=self.datamodule)
        
    def train(self):
        self.trainer.fit(model=self.model,datamodule=self.datamodule)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg)
    runner.train()

if __name__ == "__main__":
    main()

