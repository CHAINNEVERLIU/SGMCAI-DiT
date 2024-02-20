import torch
import math
from torch import nn
import numpy as np
from inspect import isfunction
from einops.layers.torch import Rearrange
from torchvision.transforms import Compose, Lambda, ToPILImage
from typing import Optional, Union
import torch.nn.functional as F
from sklearn.cluster import KMeans
import joblib
import pickle


def exists(x):
    """
    判断数值是否为空
    :param x: 输入数据
    :return: 如果不为空则True 反之则返回False
    """
    return x is not None


def default(val, d):
    """
    该函数的目的是提供一个简单的机制来获取给定变量的默认值。
    如果 val 存在，则返回该值。如果不存在，则使用 d 函数提供的默认值，
    或者如果 d 不是一个函数，则返回 d。
    :param val:需要判断的变量
    :param d:提供默认值的变量或函数
    :return:
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    """
    该函数的目的是将一个数字分成若干组，每组的大小都为 divisor，并返回一个列表，
    其中包含所有这些组的大小。如果 num 不能完全被 divisor 整除，则最后一组的大小将小于 divisor。
    :param num:
    :param divisor:
    :return:
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        """
        残差连接模块
        :param fn: 激活函数类型
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        残差连接前馈
        :param x: 输入数据
        :param args:
        :param kwargs:
        :return: f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """
    这个上采样模块的作用是将输入张量的尺寸在宽和高上放大 2 倍
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),            # 先使用最近邻填充将数据在长宽上翻倍
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),    # 再使用卷积对翻倍后的数据提取局部相关关系填充
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    """
    下采样模块的作用是将输入张量的分辨率降低，通常用于在深度学习模型中对特征图进行降采样。
    在这个实现中，下采样操作的方式是使用一个 $2 \times 2$ 的最大池化操作，
    将输入张量的宽和高都缩小一半，然后再使用上述的变换和卷积操作得到输出张量。
    由于这个实现使用了形状变换操作，因此没有使用传统的卷积或池化操作进行下采样，
    从而避免了在下采样过程中丢失信息的问题。
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        # 将输入张量的形状由 (batch_size, channel, height, width) 变换为 (batch_size, channel * 4, height / 2, width / 2)
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 对变换后的张量进行一个 $1 \times 1$ 的卷积操作，将通道数从 dim * 4（即变换后的通道数）降到 dim（即指定的输出通道数），得到输出张量。
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            mask_future=False,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)

        # Compute local map mask
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False,
                                                     device=self._scores.device)
            self._scores = self._scores.masked_fill(attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask.to(self._scores.device)
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 1024):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(F.gelu(self._linear1(x)))  # 这个地方也可改为relu


class TransformerEncoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 dff: int,
                 attention_size: int = None,
                 dropout: float = 0.3,):
        """Initialize the Encoder block"""
        super().__init__()

        self._selfAttention = MultiHeadAttention(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model, dff)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map


class InterMultiHeadAttention(nn.Module):
    """
    依据工况模态计算注意力以提取整个时间范围内的特征
    """
    def __init__(self,
                 d_model,
                 q,
                 v,
                 h,
                 attention_size=None):
        super(InterMultiHeadAttention, self).__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * self._h)
        self._W_k = nn.Linear(d_model, q * self._h)
        self._W_v = nn.Linear(d_model, v * self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h * v, d_model)

        # Score placeholder
        self._scores = None
        self._weighted_scores = None

    def forward(self,
                query,
                key,
                value,
                sampling_time_stamp,
                cluster_centers):
        """
        依据工况模态计算注意力以提取整个时间范围内的特征
        :param query: Input tensor with shape (batch_size, K, d_model) used to compute queries.
        :param key: Input tensor with shape (batch_size, K, d_model) used to compute keys.
        :param value: Input tensor with shape (batch_size, K, d_model) used to compute values.
        :param sampling_time_stamp: (batch_size, K, *)每个模态首样本的采样时间
        :param cluster_centers: (batch_size, K, *)每个模态的聚类中心
        :return:
        """
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        # print(queries.shape)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)        # [batch_size, K, K]

        # 计算采样时间之间的差距
        sampling_deter = torch.cdist(sampling_time_stamp, sampling_time_stamp)      # [batch_size, K, K]

        # 计算聚类中心之间的欧式距离差距
        cluster_centers_deter = torch.cdist(cluster_centers, cluster_centers)      # [batch_size, K, K]

        # 两类权重相加并归一化得到综合权重
        weight = torch.exp(- sampling_deter * cluster_centers_deter)
        norm_weight = weight / weight.sum(dim=1, keepdim=True)

        self._weighted_scores = self._scores * norm_weight

        # Apply sotfmax
        self._weighted_scores = F.softmax(self._weighted_scores, dim=-1)

        attention = torch.bmm(self._weighted_scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class InterConditionTransformer(nn.Module):
    """
        计算模态间的自注意力机制，主要的不同点在于给模态进行了采样间隔和模态距离加权处理
    """
    def __init__(self,
                 d_model,
                 q,
                 v,
                 h,
                 dff,
                 attention_size=None,
                 dropout=0.3):
        super(InterConditionTransformer, self).__init__()

        self._selfAttention = InterMultiHeadAttention(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model, dff)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x, sampling_time_stamp,  cluster_centers):
        """
        计算模态间的自注意力机制，主要的不同点在于给模态进行了采样间隔和模态距离加权处理
        :param x: Input tensor with shape (batch_size, K, d_model).
        :param sampling_time_stamp: (batch_size, K, *)每个模态首样本的采样时间
        :param cluster_centers: (batch_size, K, *)每个模态的聚类中心
        :return: Output tensor with shape (batch_size, K, d_model).
        """

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x,
                                sampling_time_stamp=sampling_time_stamp,
                                cluster_centers=cluster_centers)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map


class InnerConditionTransformer(TransformerEncoder):
    """
    计算模态内的自注意力机制，实现原理和原始transformer的编码器相同
    """
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 dff: int,
                 attention_size: int = None,
                 dropout: float = 0.3
                 ):
        super(InnerConditionTransformer, self).__init__(d_model, q, v, h, dff, attention_size, dropout)


class UpSamplingLayer(nn.Module):
    def __init__(self, layers):
        super(UpSamplingLayer, self).__init__()
        self.ups = nn.ModuleList([])

        for i in range(layers):
            self.ups.append(Upsample(1))

    def forward(self, x, res_list):
        for i, layer in enumerate(self.ups):
            x = layer(x) + res_list[i]

        return x


class DownSamplingLayer(nn.Module):
    def __init__(self, layers):
        super(DownSamplingLayer, self).__init__()

        self.downs = nn.ModuleList([])

        for i in range(layers):
            self.downs.append(Downsample(1))

    def forward(self, x):
        res_list = []
        for i, layer in enumerate(self.downs):
            res_list.append(x)
            x = layer(x)

        return x, res_list


class SGMCAILayer(nn.Module):
    def __init__(self, d_model, q, h, dff, layers=2, dropout=0.3):
        super(SGMCAILayer, self).__init__()

        # 定义模型结构
        self.DownSampling = DownSamplingLayer(layers)
        self.UpSampling = UpSamplingLayer(layers)

        inter_dimention = int(d_model ** 2 / (2 ** (2 * layers)))  # 下采样后得到的最终维度

        self.InterConditionTransformer = InterConditionTransformer(d_model=inter_dimention,
                                                                   q=q,
                                                                   v=q,
                                                                   h=h,
                                                                   dff=dff,
                                                                   attention_size=None,
                                                                   dropout=dropout)

        self.InnerConditionTransformer = InnerConditionTransformer(d_model=d_model,
                                                                   q=q,
                                                                   v=q,
                                                                   h=h,
                                                                   dff=dff,
                                                                   attention_size=None,
                                                                   dropout=dropout)

    def forward(self, x, sampling_time_stamp,  cluster_centers):
        """
        单层的噪声预测模型的实现代码
        :param x: 经过预处理后的输入数据 [K, M, M]
        :param sampling_time_stamp: (K, *)每个模态首样本的采样时间
        :param cluster_centers: (K, *)每个模态的聚类中心
        :return:
        """
        K = x.shape[0]
        M = x.shape[1]

        if M % 2:
            raise ValueError("输入维度必须为2的N次方，其中N为下采样层数")

        # 首先下采样输入数据
        x, res_list = self.DownSampling(x.unsqueeze(1))  # [K, 1, M/4, M/4]

        res = x.reshape((1, K, -1))  # [1, K, M*M/16]

        # 全局范围内捕捉动态特征
        x = self.InterConditionTransformer(res, sampling_time_stamp.unsqueeze(0), cluster_centers.unsqueeze(0)) + res
        # [1, K, M*M/16]

        x = x.reshape((K, 1, int(M / 4), int(M / 4)))

        res_list = res_list[::-1]
        x = self.UpSampling(x, res_list)  # [K, 1, M, M]

        x = self.InnerConditionTransformer(x.squeeze(1))  # [K, M, M]

        return x


class SGMCAINet(nn.Module):
    def __init__(self, input_size, M, num_clusters, q, h, dff, dropout,
                 layers, down_sampling_layers):
        super(SGMCAINet, self).__init__()

        self.block_length = M

        self.input_embed = nn.Linear(input_size, M)

        self.kmeans = KMeans(n_clusters=num_clusters)

        self.sampling_stamp_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(M),
            nn.GELU()
        )

        # 将原始数据集统一映射至d_model维度
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(M),
            nn.Linear(M, input_size),
            nn.GELU(),
            nn.Linear(input_size, M),
        )

        # 噪声预测模型，从下采样开始，到上采样结束
        self.predict_noise = nn.ModuleList(
            [SGMCAILayer(d_model=M, q=q, h=h, dff=dff, dropout=dropout, layers=down_sampling_layers)
             for _ in range(layers)])

        # 将隐层特征映射回原始空间
        self.hidden_transform = nn.Linear(M, input_size)

    def forward(self, x_t, t, time_stamp, kmeans_path=None, is_train=True):
        """
        工况自感知的噪声预测模型
        :param x_t: [L, K]
        :param sampling_stamp:[L, 1]采样时间
        :param t: [1] 扩散步数
        :param kmeans_path
        :param is_train
        :return:
        """
        # -------------------Step 1. 根据已有数据进行无监督聚类-------------------#
        x = x_t
        if kmeans_path is None:
            raise ValueError("必须指定KMeans模型的存储或加载路径")
        else:
            if is_train:
                # 如果是训练则需要自适应的进行无监督聚类判断工况
                self.kmeans.fit(x.detach().cpu().numpy())
                joblib.dump(self.kmeans, kmeans_path)
            else:
                self.kmeans = joblib.load(kmeans_path)
                # self.kmeans = pickle.load(kmeans_path)

        # 获取每个样本的聚类标签和聚类中心
        labels = torch.from_numpy(self.kmeans.labels_).to(x.device)  # [L, 1]
        centers = torch.from_numpy(self.kmeans.cluster_centers_).to(x.device)  # [n_clusters, N]

        # 计算每个样本距离聚类中心的距离
        x_centers = centers[self.kmeans.labels_]  # [L, N]

        # -------------------Step 2: 嵌入-------------------
        x_embed = self.input_embed(x)  # [L, M]
        time_embed = self.time_mlp(t)  # [L, M]
        x_embed = x_embed + time_embed  # [L, M]

        sampling_embed = self.sampling_stamp_embed(time_stamp)

        # -------------------Step 3: 依据工况标签分块-------------------
        # block_x: [K, M, M]|continue_lengths:[多少个持续段, 1]|block_first_index:[K, 1]
        block_x, continue_lengths, block_first_index = blocking_and_padding(x_embed, labels, self.block_length)

        # 获取到每个工况段的开始采样时间
        block_sampling_time_stamp = sampling_embed[block_first_index].to(x.device)  # [K, M]
        block_cluster_centers = x_centers[block_first_index].to(x.device)  # [K, N]

        # -------------------Step 5: 开始预测噪声-------------------
        for layer in self.predict_noise:
            block_x = layer(block_x, block_sampling_time_stamp, block_cluster_centers)

        # -------------------Step 6: 复原成原始的数据大小-------------------
        recover_block_x = self.hidden_transform(block_x)  # [K, M, N]
        recover_x = recover_block_data(recover_block_x, continue_lengths, self.block_length)  # [L, N]

        return recover_x


def blocking_and_padding(data, condition_labels, block_length):
    """
    根据输入数据的工况标签将数据集进行分块处理
    :param data: [N, M]
    :param condition_labels:[N, 1]
    :param block_length: int 分块长度
    :return: block_data [K, block_length, M]
    """

    # 先查看每个工况的持续时长
    lengths, values = count_continuous_lengths_and_values(condition_labels.cpu())

    # print(lengths)
    # print(values)

    # 首先按照工况标签将数据集切分
    wc_split_dataset = torch.split(data, lengths)

    blocks_list = []

    block_first_index = []
    first_index = 0

    for i, sub_dataset in enumerate(wc_split_dataset):
        block = split_dataset(sub_dataset, block_length)
        for j in range(block.shape[0]):
            block_first_index.append(first_index + j * block_length)
        blocks_list.append(block)
        first_index = first_index + lengths[i]

    blocks_tensor = torch.cat(blocks_list, dim=0)

    return blocks_tensor, lengths, block_first_index


def recover_block_data(block_data, length, block_length):
    start_index = 0
    recover_data = []
    for i in range(len(length)):
        block_num = length[i] // block_length  # 可以分割的整数部分
        remainder = length[i] % block_length  # 可以分割的余数部分
        # print(block_num, remainder, start_index)

        recover_data.append(block_data[start_index:start_index + block_num].reshape((-1, block_data.shape[-1])))
        start_index = start_index + block_num

        if remainder:
            recover_data.append(block_data[start_index, :remainder, :].reshape((-1, block_data.shape[-1])))
            start_index = start_index + 1

    recover_data = torch.cat(recover_data, dim=0)

    return recover_data


def count_continuous_lengths_and_values(seq):
    # 找到连续数字段的起始位置
    starts = (seq[:-1] != seq[1:]).nonzero(as_tuple=False).squeeze(1) + 1
    # 加上第一个数字的位置
    starts = torch.cat([torch.tensor([0]), starts], dim=0)

    # 找到连续数字段的结束位置
    ends = torch.cat([starts[1:], torch.tensor([len(seq)])], dim=0)

    # 计算每个数字段的长度
    lengths = ends - starts

    # 使用高级索引获取每个连续数字字段的值
    values = seq[starts]

    return lengths.numpy().tolist(), values


def smooth_condition_labels(data, condition_labels, k):
    """
    传入样本的模态标签而后进行一定程度的滤波，以消除异常点。但是目前还存在一点小bug！！
    :param data: 数据集[N, M]
    :param condition_labels: 原始分类的工况标签
    :param k: 假设长度
    :return:
    """

    if data.shape[0] != condition_labels.shape[0]:
        raise ValueError("输入数据长度和工况标签长度不一致，请检查后重新传入！")

    # print("KKK", condition_labels.shape)

    lengths, values = count_continuous_lengths_and_values(condition_labels)

    values_copy = values.clone()

    index_start = 0

    for i in range(len(lengths)):
        if i == 0 and lengths[i] < k:
            values_copy[i] = values[1]
        elif lengths[i] < k:
            values_copy[i] = values_copy[i-1]

        condition_labels[index_start:index_start + lengths[i]] = values_copy[i].repeat(lengths[i])
        index_start = index_start + lengths[i]
    return condition_labels


def split_dataset(dataset, block_length):
    """
    将二维数据集分成等长的块，如果无法等分，则在块的末尾使用整个数据集的均值填充

    Args:
        dataset: 二维数据集，形状为(N, d)
        block_length: 每个块的长度

    Returns:
        块张量，形状为(n, block_length, d)
    """
    # 计算块数
    n = dataset.shape[0] // block_length

    # 计算剩余的数据点数
    remainder = dataset.shape[0] % block_length

    # 计算整个数据集的均值
    mean = torch.mean(dataset, dim=0)

    # 使用torch.split()函数将数据集切分成n个块
    blocks = torch.split(dataset, block_length, dim=0)
    if remainder:
        # 如果剩余的数据点数不为0，就将最后一个块与均值填充的块拼接
        last_block = blocks[-1]
        last_block_length = last_block.shape[0]
        padding_size = block_length - last_block_length
        padding = mean.repeat(padding_size, 1)
        padded_block = torch.cat((last_block, padding), dim=0)
        blocks = blocks[:-1] + (padded_block,)

    # 将块列表转换为三维张量
    blocks_tensor = torch.stack(blocks)

    return blocks_tensor

