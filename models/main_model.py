import numpy as np
import torch
import torch.nn as nn
import math
from .cd2model import diff_CD2
from schedulers.sde import CD2SDE
from utils.fourier import dft, idft

class CD2_base(nn.Module):
    def __init__(self, target_dim, config, device, cd2:CD2SDE):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.cd2 = cd2
        self.time_sde = cd2.timesde
        self.freq_sde = cd2.freqsde

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CD2(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff['schedule'] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"]**0.5, config_diff["beta_end"]**0.5, self.num_steps
            ) ** 2
        elif config_diff['schedule'] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        else:
            raise NotImplementedError("Unsupported schedule.")
        
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:,:,0::2] = torch.sin(position * div_term)
        pe[:,:,1::2] = torch.cos(position * div_term)
        return pe
        
    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask
        
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
    
    def set_noise_scaling(self, max_len:int) -> None:
        G = torch.ones(max_len)
        G = 1 / math.sqrt(2) * G
        G[0] *= math.sqrt(2)
        if max_len %2 == 0:
            G[max_len//2] *= math.sqrt(2)

        self.G = G
        self.G_matrix = torch.diag(G)
        assert G.shape[0] == max_len


    # def calc_loss(
    #     self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t = -1
    # ):
    #     B, K, L = observed_data.shape
    #     if is_train != 1:
    #         t = (torch.ones(B) * set_t).long().to(self.device)
    #     else:
    #         t = torch.rand(B) * (self.time_sde.T - self.time_sde.eps) + self.time_sde.eps

    #     target_mask = (observed_mask - cond_mask).float()
    #     mean_t, std_t = self.time_sde.marginal_prob(observed_data, t)
    #     z_t = torch.randn_like(observed_data)
    #     x_t = mean_t + std_t * z_t

    #     x_input = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)

    #     diffusion_step = t
    #     x_t_score, x_f_score = self.diffmodel(x_input, side_info, diffusion_step)

    #     z_f = torch.randn_like(observed_data)
    #     x_f_input = torch.cat([cond_mask,x_t], dim=1)
    #     x_f_t = dft(x_f_input)[:,1,:,:]
    #     mean_f, std_f = self.cd2.marginal_prob(x_f_t, t)
        
    #     target_score_t = -z_t/std_t
    #     target_score_f = -z_f/std_f
    #     residual_t = (x_t_score + target_score_t) * target_mask
    #     residual_f = (x_f_score + target_score_f) * target_mask

    #     num_eval = target_mask.sum()
    #     loss_t = (residual_t ** 2).sum() / (num_eval if num_eval > 0 else 1)
    #     loss_f = (residual_f ** 2).sum() / (num_eval if num_eval > 0 else 1)
    #     return loss_t + loss_f
    # def calc_loss(
    #     self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t = -1
    # ):
    #     B, K, L = observed_data.shape
    #     if is_train == 1:
    #         t = (
    #             torch.rand(B, device=self.device) * (self.time_sde.T - self.time_sde.eps) 
    #             + self.time_sde.eps
    #         )
    #     else:
    #         t = (torch.ones(B) * set_t).to(self.device)
    #     '''
    #     mask
    #     '''
    #     target_mask = (observed_mask - cond_mask).float()
    #     '''
    #     time domain diffusion
    #     '''
    #     z_t = torch.randn_like(observed_data) # noise
    #     mean_t, std_t = self.time_sde.marginal_prob(observed_data, t)
    #     std_t = std_t.unsqueeze(-1)
    #     std_matrix_t = torch.diag_embed(std_t.squeeze(-1))
    #     inv_std_matrix_t = torch.diag_embed(1.0 / std_t.squeeze(-1))
    #     noise_t = torch.matmul(std_matrix_t, z_t)
    #     x_t = mean_t + noise_t
    #     # forward to score model (get score)
    #     x_input = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)
    #     x_t_score, x_f_score = self.diffmodel(x_input, side_info, t)
    #     target_score_t = -torch.matmul(inv_std_matrix_t, z_t)
    #     mean_f, std_f = self.cd2.marginal_prob(observed_data, t)
    #     std_f = std_f.unsqueeze(-1)
    #     std_matrix_f = torch.diag_embed(std_f.squeeze(-1))
    #     inv_std_matrix_f = torch.diag_embed(1.0 / std_f.squeeze(-1))
    #     noise_f = torch.matmul(std_matrix_f, z_t)
    #     x_f = mean_f + noise_f                             
    #     target_score_f = -torch.matmul(inv_std_matrix_f, z_t)           
    #     '''
    #     loss
    #     '''
    #     residual_t = (x_t_score + target_score_t) * target_mask
    #     residual_f = (x_f_score + target_score_f) * target_mask

    #     num_eval = target_mask.sum()
    #     loss_t = (residual_t ** 2).sum()/(num_eval if num_eval > 0 else 1)
    #     loss_f = (residual_f ** 2).sum()/(num_eval if num_eval > 0 else 1)  
    #     return loss_t + loss_f

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t = -1
    ):
        B, K, L = observed_data.shape
        if is_train:
            t = (
                torch.rand(B, device = self.device) * (self.time_sde.T - self.time_sde.eps)
                + self.time_sde.eps
            )
        else:
            t = (torch.ones(B) * set_t).to(self.device)
        '''
        mask
        '''
        target_mask = (observed_mask - cond_mask).float()
        '''
        noise
        '''
        z_t = torch.randn_like(observed_data)
        mean, std = self.cd2.marginal_prob(observed_data, t)
        var = std**2
        std_matrix = torch.diag_embed(std)
        inverse_std_matrix = torch.diag_embed(1/std)
        noise = torch.matmul(std_matrix, z_t)
        target_noise = torch.matmul(inverse_std_matrix, z_t)
        x_noisy = mean + noise
        x_input = self.set_input_to_diffmodel(x_noisy, observed_data, cond_mask)
        score_t, score_f = self.diffmodel(x_input, side_info, t)
        weighting_factor = 1.0/torch.sum(1.0/var, dim=1)
        assert weighting_factor.shape == (B,)
        losses = weighting_factor.view(-1,1,1)* torch.square(
            score_f + target_noise
        )
        residual = losses * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum()/(num_eval if num_eval > 0 else 1)
        '''
        if we want to supervise the time domain at the same time
        '''
        _, std_t = self.time_sde.marginal_prob(observed_data,t)
        var_t = std_t ** 2
        inverse_std_t_matrix = torch.diag_embed(1/std_t)
        target_noise_t = torch.matmul(inverse_std_t_matrix, z_t) 
        weighting_factor_t = 1.0/torch.sum(1.0/var_t, dim=1)
        assert weighting_factor_t.shape == (B,)
        losses_t = weighting_factor_t.view(-1,1,1)* torch.square(
            score_t + target_noise_t
        )
        residual_t = losses_t * target_mask
        loss_t = (residual_t ** 2).sum()/(num_eval if num_eval > 0 else 1)
        return loss + loss_t

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        self.cd2.set_timesteps(self.num_steps)
        for t in self.cd2.timesteps:
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        samples = torch.zeros(B, n_samples, K, L).to(self.device)
        self.cd2.set_timesteps(self.num_steps)
        for i in range(n_samples):
            # generate noisy observation for unconditional model
            x_f = self.cd2.prior_sampling((B, K, L))

            for t_scalar in self.cd2.timesteps:
                t = torch.ones(B).to(self.device) * t_scalar
                x_t = idft(x_f).to(self.device)
                x_input = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)
                score_time, score_freq = self.diffmodel(x_input, side_info, t)
                
                x_t = self.cd2.step(score_time, score_freq, t_scalar, x_t)

            x_t = cond_mask * observed_data + (1 - cond_mask) * x_t[0]
            samples[:, i] = x_t.detach()

        return samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp



class CD2_Physio(CD2_base):
    def __init__(self, config, device, cd2,target_dim=35):
        super(CD2_Physio, self).__init__(
            target_dim=target_dim,
            config=config,
            device=device,
            cd2=cd2
        )

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )