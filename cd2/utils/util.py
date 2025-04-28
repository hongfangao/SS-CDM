import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from typing import Callable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_len, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_len, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a positional encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the positional encoding should be added

        Returns:
            torch.Tensor: Tensor with an additional positional encoding
        """
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, max_len)
        pe = self.embedding(position)  # (1, max_len, d_emb)
        x = x + pe
        return x
    

class TimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_time: int, use_time_axis: bool = True):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_time, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_time, d_emb)
        self.use_time_axis = use_time_axis

    def forward(
        self, x: torch.Tensor, timesteps: torch.LongTensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        """Adds a time encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the time encoding should be added
            timesteps (torch.LongTensor): Tensor of shape (batch_size,) containing the current timestep for each sample in the batch

        Returns:
            torch.Tensor: Tensor with an additional time encoding
        """
        t_emb = self.embedding(timesteps)  # (batch_size, d_model)
        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        assert isinstance(t_emb, torch.Tensor)
        return x + t_emb


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.
    Courtesy of https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, d_model: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.d_model = d_model
        self.W = nn.Parameter(
            torch.randn((d_model + 1) // 2) * scale, requires_grad=False
        )

        self.dense = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        time_proj = timesteps[:, None] * self.W[None, :] * 2 * np.pi
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # Slice to get exactly d_model
        t_emb = embeddings[:, : self.d_model]  # (batch_size, d_model)

        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)

        projected_emb: torch.Tensor = self.dense(t_emb)

        return x + projected_emb

def Conv1d_with_init(in_channels,out_channels,kernel_size):
    layer = nn.Conv1d(in_channels,out_channels,kernel_size)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def cal_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    assert diffusion_step_embed_dim_in % 2 == 0
    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim-1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),torch.cos(_embed)), 1)
    return diffusion_step_embed

class ConstantSNREncoding(nn.Module):
    def __init__(self, gamma: float = 1.0, g: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.g_squared = g ** 2  # precompute g^2 since it's constant
        
    def forward(self, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timesteps, shape [batch_size]
            x0: initial signal, shape [batch_size, seq_len, channels]
        Returns:
            snr_condition: shape [batch_size]
        """
        # Simplified integral calculation: ∫g^2 ds from 0 to t = g^2 * t
        integral = self.g_squared * t
        
        # Calculate SNR condition
        x0_norm = torch.norm(x0, dim=(1,2))  # ||x0||^2
        snr_condition = x0_norm / (self.gamma * integral)
        
        return snr_condition
    
def cal_snr_embedding(snr_condition, snr_embed_dim):
    """
    Calculate SNR-based embedding using sinusoidal encoding
    Args:
        snr_condition: shape [batch_size], SNR values
        snr_embed_dim: dimension of the embedding
    Returns:
        snr_embed: shape [batch_size, snr_embed_dim]
    """
    assert snr_embed_dim % 2 == 0, "snr_embed_dim must be even"
    
    half_dim = snr_embed_dim // 2
    # Create frequency bands
    exponent = torch.arange(half_dim, dtype=torch.float32).cuda()
    exponent = exponent * -(math.log(10000.0) / (half_dim - 1))
    freqs = torch.exp(exponent)
    
    # Apply frequencies to SNR values
    snr_condition = snr_condition.unsqueeze(-1)  # [batch_size, 1]
    angles = snr_condition * freqs  # [batch_size, half_dim]
    
    # Create sinusoidal embedding
    sin_embed = torch.sin(angles)
    cos_embed = torch.cos(angles)
    snr_embed = torch.cat([sin_embed, cos_embed], dim=-1)
    
    return snr_embed

def get_mask_rm(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:,channel][idx] = 0
    return mask

def get_mask_mnr(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segment_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segment_index)
        mask[:,channel][s_nan[0]:s_nan[-1]+1]=0
    return mask

def get_mask_bm(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask



