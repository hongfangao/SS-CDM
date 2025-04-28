import abc
import math
from collections import namedtuple
from typing import Optional

from utils.fourier import dft, idft

import torch

SamplingOutput = namedtuple("SamplingOutput",["prev_sample"])


class SDE(abc.ABC):
    "SDE abstract class, functions are designed for a mini-batch of inputs"
    def __init__(self, fourier_noise_scaling: bool = False, eps: float = 1e-5):
        '''
        Constrcut an SDE
        Args:
            fourier_noise_scaling: if True, the noise is scaled by a parameter
            eps: a small number to avoid numerical issues
        '''
        super().__init__()
        self.noise_scaling = fourier_noise_scaling
        self.eps = eps
        self.G: Optional[torch.Tensor] = None # the noise scale matrix

    @property
    def T(self) -> float:
        "End time of SDE"
        return 1.0
    
    @abc.abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor]:
        "Compute the marginal probability p_t(x)"
        pass

    @abc.abstractmethod
    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput: 
        pass
    
    def set_timesteps(self, num_diffusion_steps:int) -> None:
        self.timesteps = torch.linspace(1.0, self.eps, num_diffusion_steps)
        self.step_size = self.timesteps[0] - self.timesteps[1]

    def set_noise_scaling(self, max_len:int) -> None:
        "Set the noise scaling matrix"
        '''
        Args:
            max_len: num of timesteps of time series, i.e., sequence length
        '''
        G = torch.ones(max_len)
        if self.noise_scaling:
            G = 1 / (math.sqrt(2)) * G
            G[0] *= math.sqrt(2)
            if max_len % 2 == 0:
                G[max_len // 2] *= math.sqrt(2)
        
        self.G = G
        self.G_matrix = torch.diag(G)
        assert G.shape[0] == max_len

    def set_timesteps(self, num_diffusion_steps:int) -> None:
        self.timesteps = torch.linspace(1.0, self.eps, num_diffusion_steps)
        self.step_size = self.timesteps[0] - self.timesteps[1]
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        "Add noise to the original samples"
        x_0 = original_samples
        # Note that the std is not used here because the noise has been scaled prior to calling the function
        mean, _ = self.marginal_prob(x_0, timestep)
        sample = mean + noise
        return sample
    
    def prior_sampling(
        self,
        shape: tuple[int, ...]
    ) -> torch.Tensor:
        # reshape the G_matrix to be (1, max_len, max_len)
        scaling_matrix = self.G_matrix.view(
            -1, self.G_matrix.shape[0], self.G_matrix.shape[1]
        )
        z = torch.randn(*shape)
        return torch.matmul(scaling_matrix, z)

'''
Here is a VPSDE with linear beta scheduler
'''
class VPScheduler(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
    ): 
        """
        Constructing Variance Preserving SDE
        Args:
            beta_min: minimum beta
            beta_max: maximum beta
            fourier_noise_scaling: whether to use Fourier noise scaling
            eps: epsilon
        """
        super().__init__(fourier_noise_scaling=fourier_noise_scaling, eps=eps)
        self.beta_0 = beta_min 
        self.beta_1 = beta_max
    
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        "Compute the marginal probability p_t(x)"
        "step 1: check whether matrix G has been initialized"
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None
        "step 2: compute the diffusion coefficient -1/2*\\int_0^t beta(s) ds"
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        ).to(x.device)
        "step 3: compute the mean and std"
        mean = (
            torch.exp(log_mean_coeff[(...,) + (None,)* len(x.shape[1:])]) * x
        )
        std = torch.sqrt(
            (1.0 - torch.exp(2.0 * log_mean_coeff.view(-1,1)))
        ) * self.G.to(
            x.device
        )
        return mean, std
    
    def get_beta(self, timestep: float) -> float:
        return self.beta_0 + timestep * (self.beta_1 - self.beta_0)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """
        beta = self.get_beta(timestep)
        assert self.G is not None
        diffusion = torch.diag_embed(
            math.sqrt(beta)* self.G
        ).to(device=sample.device)

        drift = -0.5 * beta * sample - (
            torch.matmul(diffusion*diffusion, model_output)
        )

        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output
    
class CD2SDE(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        eps: float = 1e-5
    ):
        super().__init__(fourier_noise_scaling=True, eps=eps)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.timesde = VPScheduler(
            beta_min=beta_min,
            beta_max=beta_max,
            fourier_noise_scaling=False,
            eps=eps
        )
        self.freqsde = VPScheduler(
            beta_min=beta_min,
            beta_max=beta_max,
            fourier_noise_scaling=True,
            eps=eps
        )

    def set_timesteps(self, num_diffusion_steps):
        return super().set_timesteps(num_diffusion_steps)
    
    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.timesde.G is None:
            self.timesde.set_noise_scaling(x.shape[1])
        assert self.timesde.G is not None
        if self.freqsde.G is None:
            self.freqsde.set_noise_scaling(x.shape[1])
        assert self.freqsde.G is not None
        log_mean_coeff_time = -0.25 * t**2 * (self.timesde.beta_1 - self.timesde.beta_0) - 0.5 * t * self.timesde.beta_0
        log_mean_coeff_freq = -0.25 * t**2 * (self.freqsde.beta_1 - self.freqsde.beta_0) - 0.5 * t * self.freqsde.beta_0
        log_mean_coeff = log_mean_coeff_time + log_mean_coeff_freq

        x_freq = dft(x)
        mean = (
            torch.exp(log_mean_coeff[(...,) + (None,)* len(x.shape[1:])]) * x_freq
        )

        G_f = self.freqsde.G

        std = torch.sqrt(
            1.0 - torch.exp(2.0 * log_mean_coeff.view(-1,1))
        ) * G_f.to(
            x.device
        )

        return mean, std
    
    def get_beta(self, timestep: float) -> float:
        return self.beta_0 + timestep * (self.beta_1 - self.beta_0)
    
    def step(
        self,
        model_output_t: torch.Tensor,
        model_output_f: torch.Tensor,
        timestep: float,
        sample: torch.Tensor
    ) -> SamplingOutput:
        
        '''
        sample is defined in the time domain
        '''
        sample = dft(sample)
        beta_t = self.freqsde.get_beta(timestep)
        step_size = self.step_size
        assert self.step_size > 0

        '''
        freq domain step
        '''
        drift_f = -0.5 * beta_t * sample - beta_t * model_output_f
        G_f = self.freqsde.G.to(sample.device)
        diffusion_f = torch.diag_embed(
            math.sqrt(beta_t)* G_f
        )
        z_f = torch.randn_like(sample)
        noise_f = torch.matmul(diffusion_f, z_f)
        sample = sample - drift_f * step_size + torch.sqrt(step_size) * noise_f
        sample = idft(sample)
        drift_t = -0.5 * beta_t * sample - beta_t * model_output_t
        G_t = self.timesde.G.to(sample.device)
        diffusion_t = torch.diag_embed(
            math.sqrt(beta_t)* G_t
        )
        z_t = torch.randn_like(sample)
        noise_t = torch.matmul(diffusion_t, z_t)
        sample = sample - drift_t * step_size + torch.sqrt(step_size) * noise_t
        return SamplingOutput(prev_sample=sample)
        # sample_time = idft(sample)
        # beta_t = self.timesde.get_beta(timestep)
        # assert self.timesde.G is not None
        # diffusion_t = torch.diag_embed(
        #     math.sqrt(
        #         beta_t
        #     ) * self.G
        # ).to(device=sample.device)
        # drift_t = - 0.5 * beta_t * sample_time - (
        #     torch.matmul(diffusion_t*diffusion_t, model_output_t)
        # )

        # z_t = torch.randn_like(sample_time)
        # sample_time = (
        #     sample_time 
        #     - drift_t * self.step_size
        #     + torch.sqrt(self.step_size) * torch.matmul(diffusion_t, z_t)
        # )

        # sample_freq = dft(sample_time)

        # beta_f = self.freqsde.get_beta(timestep)
        # assert self.freqsde.G is not None
        # G_f = self.freqsde.G.to(device=sample.device)
        # diffusion_f = torch.diag_embed(
        #     math.sqrt(
        #         beta_f
        #     ) * G_f
        # )

        # drift_f = -0.5 * beta_f * sample_freq - (
        #     torch.matmul(diffusion_f*diffusion_f, model_output_f)
        # )

        # z_f = torch.randn_like(sample_freq)
        # sample_freq = (
        #     sample_freq
        #     - drift_f * self.step_size
        #     + torch.sqrt(self.step_size) * torch.matmul(diffusion_f, z_f)
        # )

        # return SamplingOutput(prev_sample=sample_freq)
    
    def prior_sampling(
        self,
        shape: tuple[int, ...]
    ) -> torch.Tensor:
        scaling_matrix = self.freqsde.G_matrix.view(
            -1, self.freqsde.G_matrix.shape[0], self.freqsde.G_matrix.shape[1]
        )
        z = torch.randn(*shape)

        return torch.matmul(scaling_matrix, z)
    