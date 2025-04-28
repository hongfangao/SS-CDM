import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from linear_attention_transformer import LinearAttentionTransformer

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):
    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = localheads,
        local_attn_window_size = localwindow
    )

def get_torch_trans(heads=8,layers=1,channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model = channels,
        nhead = heads,
        dim_feedforward = channels,
        activation = 'gelu'
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    

class HighPassWindowEmbedding(nn.Module):
    """
    High-pass time embedding based on window function:
    cutoff(t) = (t * sigma^2 * gamma)^(-1/n)
    
    Replaces sin/cos in DiffusionEmbedding with frequency gating.
    """

    def __init__(
        self,
        num_steps: int,
        embedding_dim: int = 128,
        projection_dim: int = None,
        sigma: float = 0.1,
        gamma: float = 0.1,
        n: float = 1.0,
        max_freq: float = 100.0,
    ):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.embedding_dim = embedding_dim
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

        # Register embedding table (num_steps, embedding_dim)
        embedding = self._build_embedding(
            num_steps=num_steps,
            d_model=embedding_dim,
            sigma=sigma,
            gamma=gamma,
            n=n,
            max_freq=max_freq,
        )
        self.register_buffer("embedding", embedding, persistent=False)

    def forward(self, diffusion_step: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            diffusion_step: (B,) int64 tensor of diffusion step index
        Returns:
            Tensor of shape (B, projection_dim)
        """
        x = self.embedding[diffusion_step]  # (B, d_model)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(
        self, num_steps: int, d_model: int, sigma: float, gamma: float, n: float, max_freq: float
    ) -> torch.Tensor:
        """
        Precompute high-pass frequency gated embedding table.
        """
        # freq_bands: (d_model,)
        freq_bands = torch.linspace(1.0, max_freq, d_model)

        # t from 0 to num_steps-1
        t = torch.arange(num_steps).float()  # (T,)
        t_scaled = t * sigma**2 * gamma + 1e-7
        cutoff = t_scaled ** (-1 / n)  # (T,)

        # freq_weights: (T, d_model)
        freq_weights = torch.exp(-freq_bands[None, :] ** 2 / (cutoff[:, None] ** 2))

        return freq_weights  # no projection yet
    
class GaussianFourierProjection(nn.Module):
    def __init__(self, d_model:int=128, scale:float=30.0, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = d_model
        self.proj1 = nn.Linear(d_model, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)
        self.W = nn.Parameter(
            torch.randn((d_model+1)//2)*scale, requires_grad=False
        )
        self.d_model = d_model
    
    def forward(
        self, diffusion_step: torch.Tensor
    ) -> torch.Tensor:
        time_proj = diffusion_step[:,None] * self.W[None,:] * 2 * np.pi
        embeddings = torch.cat(
            [
                torch.sin(time_proj),
                torch.cos(time_proj),
            ],
            dim=-1,
        )
        t_emb = embeddings[:,:self.d_model]
        project_emb: torch.Tensor = self.proj1(t_emb)
        project_emb = F.silu(project_emb)
        project_emb = self.proj2(project_emb)
        project_emb = F.silu(project_emb)
        return project_emb

class HighPassGaussianFourierProjection(nn.Module):
    """
    High-pass time embedding based on continuous cutoff window.
    (continuous Gaussian Fourier Projection version)
    """

    def __init__(
        self,
        d_model: int = 128,
        sigma: float = 0.1,
        gamma: float = 0.1,
        n: float = 1.0,
        max_freq: float = 100.0,
        scale: float = 2 * math.pi,  # Optional overall scaling
        projection_dim: int = None,
    ):
        super().__init__()
        if projection_dim is None:
            projection_dim = d_model

        self.d_model = d_model
        self.sigma = sigma
        self.gamma = gamma
        self.n = n
        self.scale = scale

        # Define frequency bands: linspace from 1 to max_freq
        self.freq_bands = nn.Parameter(
            torch.linspace(1.0, max_freq, d_model), requires_grad=False
        )

        # Optional dense projection
        self.projection1 = nn.Linear(d_model, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, ..., D)
            timesteps: continuous t in [0,1], (B,)
            use_time_axis: if True, add time dimension

        Returns:
            Tensor: (B, 1, projection_dim) if use_time_axis else (B, projection_dim)
        """
        # Step 1: compute cutoff(t)
        t_scaled = timesteps * self.sigma**2 * self.gamma + 1e-7  # (B,)
        cutoff = t_scaled ** (-1 / self.n)  # (B,)

        # Step 2: apply Gaussian window over frequencies
        # freq_weights: (B, d_model)
        freq_weights = torch.exp(-self.freq_bands[None, :] ** 2 / (cutoff[:, None] ** 2))

        # Step 3: Fourier projection
        time_proj = timesteps[:, None] * self.freq_bands[None, :] * self.scale  # (B, d_model)
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)  # (B, 2*d_model)

        # Step 4: apply frequency gating
        gated_embeddings = embeddings[:, :self.d_model] * freq_weights  # (B, d_model)


        # Step 5: Nonlinear projection
        out = self.projection1(gated_embeddings)
        out = F.silu(out)
        out = self.projection2(out)
        out = F.silu(out)

        return out
