import numpy as np
import torch


def get_rand_mask(observed_mask):
    """
    在可观测点的基础上再随机屏蔽一些点作为补全目标，用于自监督学习中使用
    :param observed_mask: 原始数据是否可观测的标记矩阵
    :return: 屏蔽后的条件矩阵
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    for i in range(len(observed_mask)):
        sample_ratio = np.random.rand()  # missing ratio
        # sample_ratio = 0.5
        num_observed = observed_mask[i].sum().item()
        num_masked = round(num_observed * sample_ratio)
        rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask





