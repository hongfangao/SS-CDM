from typing import Optional

import torch
from tqdm import tqdm

from cd2.models.score_models import ScoreModule, CD2
from cd2.utils.fdataclasses import DiffusableBatch
from cd2.schedulers.sde import SDE

class DiffusionSampler:
    def __init__(
        self,
        CD2ScoreModel: CD2,
        sample_batch_size: int,
    ) -> None:
        self.model = CD2ScoreModel
        self.noise_scheduler = self.model.noise_scheduler
        self.sample_batch_size = sample_batch_size
        self.in_channels = self.model.in_channels
        self.seq_len = self.model.seq_len

    def reverse_diffusion_step(self, batch: DiffusableBatch) -> torch.Tensor:
        
        X = batch.X
        timesteps = batch.timesteps

        assert timesteps is not None and timesteps.size(0) == len(batch)
        assert torch.min(timesteps) == torch.max(timesteps)

        '''
        check this
        '''
        score = self.model(batch)
        output = self.noise_scheduler.step(
            model_output=score, timesteps=timesteps[0].item(), sample=X
        )

        X_prev = output.prev_sample
        assert isinstance(X_prev, torch.Tensor)

        return X_prev
    
    def sample(
        self,
        num_samples: int,
        num_diffusion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        self.model.eval()
        num_diffusion_steps = (
            self.model.num_training_steps
            if num_diffusion_steps is None
            else num_diffusion_steps
        )
        self.noise_scheduler.set_timesteps(num_diffusion_steps)

        all_samples = []
        num_batches = max(1,num_samples//self.sample_batch_size)

        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc="Sampling",
                unit="batch",
                leave=False,
                colour="blue"
            ):
                batch_size = min(
                    num_samples - batch_idx *self.sample_batch_size,
                    self.sample_batch_size,
                )

                X = self.sample_prior(batch_size)
                for t in tqdm(
                    self.noise_scheduler.timesteps,
                    desc="Diffusion",
                    unit="step",
                    leave=False,
                    colour="green",
                ):
                    timesteps = torch.full(
                        (batch_size,),
                        t,
                        dtype=(
                            torch.long if isinstance(t.item(), int) else torch.float
                        ),
                        device=self.model.device,
                        requires_grad=False,
                    )
                    batch = DiffusableBatch(X=X,y=None,timesteps=timesteps)
                    X = self.reverse_diffusion_step(batch)
                all_samples.append(X.cpu())
        return torch.cat(all_samples, dim=0)
    
    def sample_prior(self, batch_size: int) -> torch.Tensor:
        if isinstance(self.noise_scheduler, SDE):
            X = self.noise_scheduler.prior_sampling(
                (batch_size, self.seq_len, self.in_channels)
            ).to(self.model.device)
        else:
            raise NotImplementedError("Unrecongnized Scheduler")
        
        assert isinstance(X, torch.Tensor)
        return X
            