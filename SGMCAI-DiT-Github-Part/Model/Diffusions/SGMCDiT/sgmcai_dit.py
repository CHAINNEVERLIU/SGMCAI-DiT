import torch
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm

from Model.Diffusions.Base.class_declaration import DiffusionBase
from Model.Diffusions.Base.variance_schedule import VarianceSchedule
from Model.Diffusions.Base.tools import *

from Tools.mask_tools import get_rand_mask
from Tools.train_network_helper import TrainBase

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class SGMCAIDiT(DiffusionBase):
    def __init__(self,
                 schedule_name="linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super(SGMCAIDiT, self).__init__()

        # 噪声模型初始化
        self.denoise_model = denoise_model

        # 方差生成
        variance_schedule_func = VarianceSchedule(
                                                schedule_name=schedule_name,
                                                beta_start=beta_start,
                                                beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func.get_schedule(timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None) -> Tensor:
        """
        Forward Process: x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        :param x_start: x_0 -> [B, L, K]
        :param t: current time step t
        :param noise: z_t -> [B, L, K]
        :return:
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 提取到t时刻的系数
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # 返回真实的x_t
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t,
                     noise=None,
                     loss_type="l2",
                     observed_mask=None,
                     cond_mask=None,
                     time_stamp=None,
                     kmeans_path=None,
                     is_train=True) -> Tensor:
        """
        计算t时间步的噪声预测误差
        :param x_start: x_0 -> [B, L, K]
        :param t: current time step t
        :param noise: z_t -> [B, L, K]
        :param loss_type: ['l1', 'l2', 'huber']
        :param observed_mask: [B, L, K], observed is 1, otherwise is 0
        :param cond_mask: [B, L, K], Manually block some again based observed_mask
        :param time_stamp: sampling time stamp -> [B, L]
        :param kmeans_path: save or load path of kmeans model
        :param is_train: True or False
        :return: loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if cond_mask is None:
            target_mask = 1 - observed_mask
        else:
            target_mask = observed_mask - cond_mask

        x_t_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # add noise to missing parts
        x_t = x_start * cond_mask + x_t_noisy * (1 - cond_mask)

        predicted_noise = self.denoise_model(
            x_t=x_t, t=t, time_stamp=time_stamp, kmeans_path=kmeans_path, is_train=is_train
        )

        # 只计算手动屏蔽部分的噪声预测误差
        noise_target = noise * target_mask

        if loss_type == 'l1':
            loss = F.l1_loss(noise_target, predicted_noise * target_mask)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise_target, predicted_noise * target_mask)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise_target, predicted_noise * target_mask)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index, observed_mask, time_stamp, kmeans_path) -> Tensor:
        """
        Inverse Procees
        :param x: x_t
        :param t:
        :param t_index:
        :param observed_mask:
        :param time_stamp:
        :param kmeans_path:
        :return:
        """
        # x是上一时刻得到的x, target_mask中缺失的位置为1, 没缺失的位置为0
        target_mask = 1 - observed_mask

        # 提取t时刻的系数
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        predicted_noise = self.denoise_model(
            x_t=x, t=t, time_stamp=time_stamp, kmeans_path=kmeans_path, is_train=False
        )

        # 计算添加了噪声的地方的均值
        model_mean = sqrt_recip_alphas_t * (
                x - (betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        ) * target_mask

        if t_index == 0:
            # return model_mean * missing_map + observed_x
            return model_mean + x * observed_mask
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)

            # 只在有缺失的位置加噪
            noise = torch.randn_like(x) * target_mask
            # Algorithm 2 line 4:
            # return (model_mean + torch.sqrt(posterior_variance_t) * noise) * missing_map + observed_x
            return model_mean + torch.sqrt(posterior_variance_t) * noise + x * observed_mask

    @torch.no_grad()
    def p_sample_loop(self, observed_x, observed_mask, time_stamp, kmeans_path) -> Tensor:
        # target_mask中缺失的位置为1, 没缺失的位置为0
        target_mask = 1 - observed_mask

        device = next(self.denoise_model.parameters()).device

        b = observed_x.shape[0]

        # start from pure noise (for each example in the batch)
        sample = observed_x + torch.randn(observed_x.shape, device=device) * target_mask

        # 用于存储在不同步数得到的暂存图像
        samples = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step',
                      total=self.timesteps, disable=True, end='\r'):
            sample = self.p_sample(x=sample,
                                   t=torch.full((b,), i, device=device, dtype=torch.long),
                                   t_index=i,
                                   observed_mask=observed_mask,
                                   time_stamp=time_stamp,
                                   kmeans_path=kmeans_path
                                   )
            samples.append(sample.cpu().numpy())

        return samples

    @torch.no_grad()
    def sample(self, observed_x, observed_mask, time_stamp, kmeans_path) -> Tensor:
        return self.p_sample_loop(observed_x, observed_mask, time_stamp, kmeans_path)

    def forward(self, x_start, t, observed_mask, time_stamp, kmeans_path):
        cond_mask = get_rand_mask(observed_mask)
        return self.compute_loss(x_start=x_start,
                                 t=t,
                                 observed_mask=observed_mask,
                                 cond_mask=cond_mask,
                                 time_stamp=time_stamp,
                                 kmeans_path=kmeans_path)

    def impute(self, sampling_times, observed_x, observed_mask, time_stamp, kmeans_path):
        imputed_data = []
        for i in tqdm(range(sampling_times), desc='sampling loop', total=sampling_times, disable=False):
            samples = self.sample(observed_x, observed_mask, time_stamp, kmeans_path)[-1]
            imputed_data.append(samples)

        imputed_data = np.stack(imputed_data, axis=0)
        # imputed_data_mean = np.mean(imputed_data, axis=0)
        imputed_data_mean = np.median(imputed_data, axis=0)

        return imputed_data_mean, imputed_data


class SGMCAIDiTTrain(TrainBase):
    def __init__(self,
                 timesteps,
                 epochs,
                 train_loader,
                 optimizer,
                 loss_function,
                 lr,
                 device,
                 patience=None,
                 val_loader=None,
                 val_epochs=1,
                 checkpoint_path=None,
                 train_log_path=None):
        super(SGMCAIDiTTrain, self).__init__(
                                            epochs,
                                            train_loader,
                                            optimizer,
                                            loss_function,
                                            lr,
                                            device,
                                            patience,
                                            val_loader,
                                            val_epochs,
                                            checkpoint_path,
                                            train_log_path)
        self.timesteps = timesteps

    def _get_train_batch_loss(self, model, *args, **kwargs):
        train_batch_loss = []
        for step, (data, observed_mask, time_df) in enumerate(self.train_loader):
            # data :[1, L, K]，将整个数据集视为一个batch，提取全局特征
            batch_size = data.shape[0]
            observed_data = data * observed_mask

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

            loss = model(x_start=observed_data, t=t, observed_mask=observed_mask,
                         time_stamp=time_df, kmeans_path=kwargs["kmeans_path"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_batch_loss.append(loss.cpu().detach().item())

        train_batch_loss = np.average(train_batch_loss)

        return train_batch_loss

    def _get_val_batch_loss(self, model, *args, **kwargs):
        average_impute_loss = None
        for val_step, (val_data, val_observed_mask, val_time_df) in enumerate(self.train_loader):
            imputed_data_mean, imputed_data = model.impute(observed_x=val_data * val_observed_mask,
                                                           observed_mask=val_observed_mask,
                                                           time_stamp=val_time_df,
                                                           sampling_times=kwargs["sampling_times"],
                                                           kmeans_path=kwargs["kmeans_path"])

            target_mask = 1 - val_observed_mask
            target_mask = target_mask.cpu().numpy()

            average_impute_loss = np.sum(
                np.square((imputed_data_mean - val_data.cpu().numpy()) * target_mask)) / np.sum(target_mask)

        return average_impute_loss



