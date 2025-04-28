import logging
from pathlib import Path

import hydra
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cd2.dataloaders.datamodules import DataModule
from cd2.models.score_models import ScoreModule, CD2
from cd2.sampling.metrics import MetricCollection
from cd2.sampling.sampler import DiffusionSampler
from cd2.utils.extraction import dict_to_str, get_best_ckpt, get_model_type

class SamplingRunner:
    def __init__(
        self,
        cfg: DictConfig
    ) -> None:
        self.random_seed: int = cfg.random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        
        logging.info(
            f"Start sampling using config:\n{dict_to_str(cfg)}"
        )

        self.model_path = Path(cfg.model_path)
        self.model_id = cfg.model_id
        
        self.save_dir = self.model_path / self.model_id
        OmegaConf.save(config=cfg, f=self.save_dir/"sample_config.yaml")

        train_cfg = OmegaConf.load(self.save_dir/"train_config.yaml")
        self.datamodule: DataModule = instantiate(train_cfg.datamodule)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.num_samples: int = cfg.num_samples 
        self.num_diffusion_steps: int = cfg.num_diffusion_steps

        best_ckpt_path = get_best_ckpt(self.save_dir/"checkpoints")
        model_type = get_model_type(train_cfg)
        self.model = model_type.load_from_checkpoint(
            checkpoint_path=best_ckpt_path
        )
        if torch.cuda.is_available():
            self.model.to(device=torch.device("cuda"))

        sampler_partial = instantiate(cfg.sampler)
        self.sampler: DiffusionSampler = sampler_partial(score_model=self.model)

        metrics_partial = instantiate(cfg.metrics)
        self.metrics: MetricCollection = metrics_partial(
            original_samples = self.datamodule.X_train
        )

    def sample(self) -> None:
        X = self.sampler.sample(
            num_samples=self.num_samples, num_diffusion_steps=self.num_diffusion_steps
        )
        if self.datamodule.standardize:
            feature_mean, feature_std = self.datamodule.feature_mean_and_std
            X = X * feature_std + feature_mean
        
        results = self.metrics(X)
        logging.info(f"Metrics:\n{dict_to_str(results)}")

        logging.info(f"Saving samples and metrics to {self.save_dir}.")
        yaml.dump(
            data = results,
            stream = open(self.save_dir/"results.yaml","w")
        )
        torch.save(X,self.save_dir/"samples.pt")

@hydra.main(version_base=None, config_path="conf",config_name="sample")
def main(cfg: DictConfig) -> None:
    runner = SamplingRunner(cfg)
    runner.sample()

if __name__ == "__main__":
    main()