from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim 
from diffusers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from mamba_ssm.modules.mamba2 import Mamba2

from cd2.schedulers.sde import SDE
from cd2.utils.fdataclasses import DiffusableBatch
from cd2.utils.util import PositionalEncoding, TimeEncoding, GaussianFourierProjection
from cd2.utils.losses import get_sde_loss_fn
from cd2.utils.fourier import dft, idft
from cd2.utils.util import Conv1d_with_init, cal_diffusion_step_embedding, cal_snr_embedding, ConstantSNREncoding
from cd2.utils.util import get_mask_bm, get_mask_mnr, get_mask_rm

import math

class ScoreModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        noise_scheduler: SDE,
        fourier_noise_scaling: bool = True,
        d_model: int = 64,
        num_layers: int = 3,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
        likelihood_weighting: bool = False,
    ) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.noise_scheduler = noise_scheduler
        self.num_warm_up_steps = num_training_steps // 10
        self.lr_max = lr_max
        self.scale_noise = fourier_noise_scaling

        self.likelihood_weighting = likelihood_weighting
        self.training_loss_fn, self.validation_loss_fn = self.set_loss_fn()

        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        self.time_encoder = nn.Linear(d_model,seq_len)
        self.embedder = nn.Linear(in_channels, d_model)
        self.unembdder = nn.Linear(d_model, in_channels)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model = d_model, nhead = 1, batch_first = True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer = transformer_layer, num_layers = num_layers
        )

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.seq_len,
            self.in_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        X = self.embedder(X)
        X = self.pos_encoder(X)
        X = self.time_encoder(X, timesteps)
        X = self.backbone(X)
        X = self.unembdder(X)
        assert isinstance(X, torch.Tensor)

        return X

    def training_step(
            self, batch: DiffusableBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )

    def validation_step(
        self, batch: DiffusableBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warm_up_steps,
            num_training_steps=self.trainer.max_steps,
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}    
    
    def set_loss_fn(
        self,
    ) -> tuple[
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
    ]:
        if isinstance(self.noise_scheduler, SDE):
            training_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=True,
                likelihood_weighting=self.likelihood_weighting
            )
            validation_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=False,
                likelihood_weighting=self.likelihood_weighting
            )

            return training_loss_fn, validation_loss_fn
        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set loss function."
            )
        
    def set_time_encoder(self) -> TimeEncoding | GaussianFourierProjection:
        if isinstance(self.noise_scheduler, SDE):
            return GaussianFourierProjection(d_model=self.d_model)

        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set time encoder."
            )

class ResidualBlock(pl.LightningModule):
    def __init__(
        self, 
        in_channels,
        num_channels,
        embedding_dim,
        seq_len,
        expand,
        headdim
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = num_channels
        self.seq_len = seq_len
        '''
        input embedding
        '''
        self.input_proj = Conv1d_with_init(in_channels, num_channels, 1)
        self.input_ssm = Mamba2(d_model=num_channels, expand=expand, headdim=headdim)
        self.diffusion_proj = nn.Linear(embedding_dim, num_channels)
        '''
        cond embedding
        '''
        self.cond_proj = Conv1d_with_init(2*in_channels,2*num_channels,1)
        self.cond_ssm = Mamba2(d_model=2*num_channels, expand=expand, headdim=headdim)
        self.time_proj = Conv1d_with_init(num_channels,2*num_channels,1)
        self.time_ssm = Mamba2(d_model=num_channels, expand=expand, headdim=headdim) # check dimension of time_ssm
        self.last_ssm = Mamba2(d_model=2*num_channels, expand=expand, headdim=headdim)
        self.skip_proj = Conv1d_with_init(num_channels, num_channels, 1)
        self.res_proj = Conv1d_with_init(num_channels, in_channels, 1)

    def forward(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.in_channels
        assert L == self.seq_len
        '''
        input embedding
        '''
        h = self.input_proj(h)
        h = self.input_ssm(h)
        '''
        diffusion step embedding
        '''
        diffusion_step_embed = self.diffusion_proj(diffusion_step_embed)
        diffusion_step_embed = diffusion_step_embed.view([B,self.time_channels,1])
        h = h + diffusion_step_embed
        h = self.time_ssm(h)
        h = self.time_proj(h)
        '''
        cond embedding
        '''
        assert cond is not None
        cond = self.cond_proj(cond)
        cond = self.cond_ssm(cond)
        h += cond
        h = self.last_ssm(h)
        out = torch.tan(h[:,self.time_channels:,:])*torch.sigmoid(h[:,0:self.time_channels,:])
        res = self.res_proj(out)
        assert res.shape == x.shape
        skip = self.skip_proj(out)
        return (x+res)*math.sqrt(0.5), skip
    
class TimeModule(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        num_channels,
        embedding_dim,
        seq_len,
        expand,
        headdim,
        num_layers,
        mask:str,
        mask_k:int,
    ) -> None:
        super().__init__()
        self.out_proj = Conv1d_with_init(num_channels, in_channels, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels = in_channels,
                    num_channels = num_channels,
                    embedding_dim = embedding_dim,
                    seq_len = seq_len,
                    expand = expand,
                    headdim = headdim
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.diffusion_embedding_dim = embedding_dim
        self.mask = mask
        self.mask_k = mask_k
        self.save_hyperparameters()
    
    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.seq_len,
            self.in_channels,
        ),f"X has wrong shape, should be {(X.size(0), self.seq_len, self.in_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)
        diffusion_step_embedding = cal_diffusion_step_embedding(timesteps, self.diffusion_embedding_dim)
        assert self.mask in ['bm', 'mnr', 'rm']
        assert self.domain in ['time', 'freq']
        if self.mask == 'bm':
            mask = get_mask_bm(X,self.mask_k)
            cond = X * mask
            cond = torch.cat([cond, mask.float()], dim=1)
        elif self.mask == 'rm':
            mask = get_mask_rm(X,self.mask_k)
            cond = X * mask
            cond = torch.cat([cond, mask.float()], dim=1)
        elif self.mask == 'mnr':
            mask = get_mask_mnr(X,self.mask_k)
            cond = X * mask
            cond = torch.cat([cond, mask.float()], dim=1)
        else:
            raise NotImplementedError        
        
        skip = 0
        h = X
        for n in range(self.num_layers):
            h, skip_n = self.residual_layers[n]((h, cond, diffusion_step_embedding))
            skip += skip_n
        x = skip/math.sqrt(self.num_layers)
        x = self.out_proj(x)
        assert isinstance(x, torch.Tensor)
        return x

class FreqModule(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        num_channels,
        embedding_dim,
        seq_len,
        expand,
        headdim,
        num_layers,
    ) -> None:
        super().__init__()
        self.out_proj = Conv1d_with_init(num_channels, in_channels, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels = in_channels,
                    num_channels = num_channels,
                    embedding_dim = embedding_dim,
                    seq_len = seq_len,
                    expand = expand,
                    headdim = headdim
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.diffusion_embedding_dim = embedding_dim
        self.SNR = ConstantSNREncoding()
        self.save_hyperparameters()
    
    def forward(self, batch: DiffusableBatch, cond: torch.Tensor) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.seq_len,
            self.in_channels,
        ),f"X has wrong shape, should be {(X.size(0), self.seq_len, self.in_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)
        timesteps = self.SNR(timesteps)
        diffusion_step_embedding = cal_snr_embedding(timesteps, self.diffusion_embedding_dim)
        
        skip = 0
        h = X
        for n in range(self.num_layers):
            h, skip_n = self.residual_layers[n]((h, cond, diffusion_step_embedding))
            skip += skip_n
        x = skip/math.sqrt(self.num_layers)
        x = self.out_proj(x)
        assert isinstance(x, torch.Tensor)
        return x

class CD2(pl.LightningModule):
    def __init__(
        self,
        in_channels_time,
        time_channels,
        embedding_dim_time,
        time_seq_len,
        expand_t,
        headdim_t,
        num_layers_t,
        in_channels_freq,
        freq_channels,
        embedding_dim_freq,
        freq_seq_len,
        expand_f,
        headdim_f,
        num_layers_f,
        mask,
        mask_k,
        num_training_steps,
        lr_max,
        num_warmup_steps,
        noise_scheduler: SDE,
        likelihood_weighting: bool,
    ):
        '''
        due to joint training, the num_training_steps is the same for both time and freq
        '''
        super().__init__()
        self.time_module = TimeModule(
            in_channels=in_channels_time,
            num_channels=time_channels,
            embedding_dim=embedding_dim_time,
            seq_len=time_seq_len,
            expand=expand_t,
            headdim=headdim_t,
            num_layers=num_layers_t,
            mask=mask,
            mask_k=mask_k
        )
        self.freq_module = FreqModule(
            in_channels_freq,
            freq_channels,
            embedding_dim_freq,
            freq_seq_len,
            expand_f,
            headdim_f,
            num_layers_f,
            num_training_steps
        )
        self.lr_max = lr_max
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.noise_scheduler = noise_scheduler
        self.likelihood_weighting = likelihood_weighting
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        timesteps = batch.timesteps
        y = batch.y if batch.y is not None else None
        X = self.time_module(batch)
        X = dft(X)
        y = dft(y) if y is not None else None
        batch = DiffusableBatch(X,y,timesteps)
        X = self.freq_module(batch, cond=X)
        X = idft(X)
        assert isinstance(X, torch.Tensor)
        return X
    
    def training_step(
        self,
        batch: DiffusableBatch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)
        self.log_dict(
            {"train/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=True
        )
        return loss

    def validation_step(
        self,
        batch: DiffusableBatch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False   
        )
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        lr_scheduler_config = {"scheduler":lr_scheduler, "intervel":"step"}
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler_config}
    
    def set_loss_fn(
        self,
    ) -> tuple[
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
        Callable[[nn.Module, DiffusableBatch], torch.Tensor]
    ]:
        if isinstance(self.noise_scheduler,SDE):    
            training_loss_fn = get_sde_loss_fn(
                scheduler = self.noise_scheduler,
                train = True,
                likelihood_weighting = self.likelihood_weighting
            )
            validation_loss_fn = get_sde_loss_fn(
                scheduler = self.noise_scheduler,
                train = False,
                likelihood_weighting = self.likelihood_weighting
            )
        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set loss function."
            )           