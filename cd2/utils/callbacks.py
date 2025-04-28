import pytorch_lightning as pl
import torch 

from cd2.dataloaders.datamodules import DataModule
from cd2.models.score_models import ScoreModule
from cd2.sampling.metrics import Metric, MetricCollection
from cd2.sampling.sampler import DiffusionSampler

class SamplingCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int,
        sample_batch_size: int,
        num_samples: int,
        num_diffusion_steps: int,
        metrics: list[Metric],
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sample_batch_size = sample_batch_size
        self.num_samples = num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.metrics = metrics
        self.datamodule_initialized = False

    def setup_datamodule(self, datamodule: DataModule) -> None:
        self.standardize = datamodule.standardize
        self.feature_mean, self.feature_std = datamodule.feature_mean_and_std
        self.metric_collection = MetricCollection(
            metrics = self.metrics,
            original_samples = datamodule.X_train,
            include_baselines = False
        )
        self.datamodule_initialized = True
    
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: ScoreModule,
    ) -> None:
        self.sampler = DiffusionSampler(
            score_model = pl_module,
            sample_batch_size = self.sample_batch_size
        )
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            or trainer.current_epoch + 1 == trainer.max_epochs
        ):
            X = self.sample()
            results = self.metric_collection(X)
            results = {f"metrics/{key}": value for key, value in results.items()}
            pl_module.log_dict(results, on_step=False, on_epoch=True)

    def sample(self) -> torch.Tensor:
        assert self.datamodule_initialized, (
            "The datamodule has not been initialized. "
            "Please call `setup_datamodule` before sampling."
        )
        X = self.sampler.sample(
            num_samples=self.num_samples,
            num_diffusion_steps=self.num_diffusion_steps,
        )

        if self.standardize:
            X = X * self.feature_std + self.feature_mean
        
        assert(isinstance(X,torch.Tensor))

        return X