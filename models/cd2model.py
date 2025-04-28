import torch
import torch.nn as nn
import math

import torch.nn.functional as F

from utils.util import get_linear_trans, get_torch_trans, Conv1d_with_init
from utils.util import GaussianFourierProjection, HighPassGaussianFourierProjection
from utils.fourier import dft 


class diff_CD2(nn.Module):
    def __init__(
        self,
        config: dict,
        inputdim: int = 2,
    ):
        super().__init__()

        # hyperparameters
        self.channels = config["channels"]

        self.time_diffusion_embedding = GaussianFourierProjection(
            d_model=config["time_diffusion_embedding_dim"],
        )
        self.freq_diffusion_embedding = HighPassGaussianFourierProjection(
            d_model=config["freq_diffusion_embedding_dim"],
        )
        '''
        time diffusion projections
        '''
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        '''
        freq diffusion projections
        '''
        self.freq_input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.freq_out_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.freq_out_projection2 = Conv1d_with_init(self.channels, 1, 1)
        
        nn.init.zeros_(self.output_projection2.weight)
        nn.init.zeros_(self.freq_out_projection2.weight)

        self.residual_layers_time = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim = config['side_dim'],
                    channels = self.channels,
                    diffusion_embedding_dim = config['time_diffusion_embedding_dim'],
                    nheads = config['nheads_time'],
                    is_linear = config['is_linear_time']
                )
            ]
        )

        self.residual_layers_freq = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim = config['side_dim'],
                    channels = self.channels,
                    diffusion_embedding_dim = config['freq_diffusion_embedding_dim'],
                    nheads = config['nheads_freq'],
                    is_linear = config['is_linear_freq'],
                )
            ]
        )


    def forward(self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_step: float):
        '''
        here x contains mask matrix, so inputdim is 2
        diffusion_step implementation
        '''
        B, inputdim, K, L = x.shape
        mask = x[:,0,:,:].unsqueeze(1)
        x = x.reshape(B, inputdim, K*L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        
        diffusion_emb = self.time_diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers_time:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers_time))
        x = x.reshape(B, self.channels, K*L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        
        x_t = x.reshape(B, K, L)
        x = x.reshape(B, 1, K, L)

        x = torch.cat([mask,x], dim=1)
        x = dft(x)

        x = x.reshape(B, inputdim, K*L)
        x = self.freq_input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        freq_diffusion_emb = self.freq_diffusion_embedding(diffusion_step)
        
        skip_f = []
        for layer in self.residual_layers_freq:
            x, skip_connection_f = layer(x, cond_info, freq_diffusion_emb)
            skip_f.append(skip_connection_f)

        x = torch.sum(torch.stack(skip_f), dim=0) / math.sqrt(len(self.residual_layers_freq))
        x = x.reshape(B, self.channels, K*L)
        x = self.freq_out_projection1(x)
        x = F.relu(x)
        x = self.freq_out_projection2(x)
        x = x.reshape(B, K, L)

        assert isinstance(x, torch.Tensor)
        assert isinstance(x_t, torch.Tensor)

        return x_t, x
        
class ResidualBlock(nn.Module):
    def __init__(self, 
            side_dim: int, 
            channels: int, 
            diffusion_embedding_dim: int, 
            nheads: int, 
            is_linear: bool=False
        ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads,layers=1,channels=channels)

    def forward_time(self, 
            y: torch.Tensor, 
            base_shape: tuple[int, int, int, int]
        ):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B*K, channel, L)
        if self.is_linear:
            y = self.time_layer(y.permute(0,2,1)).permute(0,2,1)
        else:
            y = self.time_layer(y.permute(2,0,1)).permute(1,2,0)
        
        y = y.reshape(B, K, channel, L).permute(0,2,3,1).reshape(B, channel, K*L)

        return y
    
    def forward_feature(self, 
            y: torch.Tensor, 
            base_shape: tuple[int, int, int, int]
        ):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B*K, channel, L)
        if self.is_linear:
            y = self.feature_layer(y.permute(0,2,1)).permute(0,2,1)
        else:
            y = self.feature_layer(y.permute(2,0,1)).permute(1,2,0)
        
        y = y.reshape(B, K, channel, L).permute(0,2,3,1).reshape(B, channel, K*L)

        return y
    
    def forward(self, 
            x: torch.Tensor, 
            cond_info: torch.Tensor, 
            diffusion_emb: torch.Tensor,
        ):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K*L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1) # (B, C, 1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K*L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y,2,dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual)/math.sqrt(2.0), skip

