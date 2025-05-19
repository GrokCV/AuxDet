import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from einops import rearrange

from mmdet.models.utils import DMLPAttention, FCResLayer
from mmdet.structures import DetDataSample


@MODELS.register_module()
class AuxFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            start_level: int = 0,
            end_level: int = -1,
            add_extra_convs: Union[bool, str] = False,
            relu_before_extra_convs: bool = False,
            no_norm_on_lateral: bool = False,
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = None,
            act_cfg: OptConfigType = None,
            upsample_cfg: ConfigType = dict(mode='nearest'),
            init_cfg: MultiConfig = dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # （多层级）轻量级边缘增强模块
        self.edge_convs = nn.ModuleList([
            EdgeConvSep(out_channels, out_channels)
            for _ in range(2)
        ])

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # 浅层下采样，以对齐深层
        self.downsamples = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=8, padding=1, groups=out_channels // 8),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=4, padding=1, groups=out_channels // 8),
            ])

        # 图像特征-辅助特征 动态融合，产生相关调制参数
        self.dynamic_atts = nn.ModuleList([
            DMLPAttention(in_channels_f1=out_channels) for _ in range(2)
        ])
        # 平均池化 语义补偿特征
        self.sem_gap = nn.AdaptiveAvgPool2d(1)
        # 空间调制
        self.sm = SpatialModulation()

        # 高层语义补偿后的辅助特征处理
        self.meta_process_with_sem = MetaFeatureProcessorWithSem(channel_outs=out_channels // 2)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels//2, out_channels//4),
            nn.ReLU(),
            nn.Linear(out_channels//4, 1),
            nn.Sigmoid()  # 限制权重范围在 (0,1)
        )

    def forward(self, inputs: Tuple[Tensor], meta_inf: List[DetDataSample]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.
                Shape: (Batch_size, Channels, H, W).
            meta_inf (List[DataSample]): Batch data samples containing meta information of images.

        Returns:
            tuple[Tensor]: Output feature maps, each is a 4D-tensor.
                Shape: (Batch_size, Out_channels, H', W').
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        alphas = []
        for idx in range(2):
            # step 1. 获取补偿特征并降维
            sem_comp_fea = self.downsamples[idx](laterals[idx]) - laterals[-1]
            sem_inf = self.sem_gap(sem_comp_fea).flatten(1)
            # step 2. 融合 元数据特征及补偿特征，得到最终辅助特征
            all_aux_fea = self.meta_process_with_sem(
                meta_inf, inputs[0].dtype, inputs[0].device, sem_inf)

            # 为当前层计算 ALPHA 并保存，用于自适应调整后续边缘特征增强
            current_alpha = self.mlp(all_aux_fea).view(-1, 1, 1, 1)
            alphas.append(current_alpha)  # 保存到列表

            # 注：通道-空间 调制 并行展开
            B, C, _, _ = laterals[idx].shape
            # step 3. 动态融合 最终辅助特征与当前层的图像特征，得到调制参数
            modulation_weights = self.dynamic_atts[idx](laterals[idx], all_aux_fea)
            # step 4. 分解得到 通道调制参数 和 空间调制参数
            channel_modulation_weights, spatial_modulation_weights = modulation_weights[:, :C].sigmoid(), modulation_weights[:, C:].sigmoid()

            # step 5. 分别进行 逐样本得 通道调制 和 空间调制
            c = laterals[idx] * channel_modulation_weights.reshape(B, C, 1, 1) + laterals[idx]
            s = self.sm(laterals[idx], spatial_modulation_weights)

            # step 6. 融合调制后的特征
            laterals[idx] = laterals[idx] + s + c

        # 进行边缘增强
        edge_features = [
            edge_conv(laterals[i])
            for i, edge_conv in enumerate(self.edge_convs)
        ]

        # 边缘增强后 与原特征 融合
        for i in range(len(edge_features)):  # 只遍历前两个
            laterals[i] = laterals[i] + alphas[i] * edge_features[i]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # self.fpn_convs[i](laterals[i]) for i in range(self.num_outs)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # return tuple(outs)
        return tuple(outs[:2])


class MetaFeatureProcessorWithSem(nn.Module):
    def __init__(self, channel_outs=256):
        super().__init__()
        # 视角编码 MLP：3 (one-hot) -> 64
        self.view_mlp = nn.Sequential(
            nn.Linear(3, channel_outs // 4),
            nn.ReLU(),
            nn.BatchNorm1d(channel_outs // 4)
        )

        # 波段编码 MLP：3 (one-hot) -> 64
        self.band_mlp = nn.Sequential(
            nn.Linear(3, channel_outs // 4),
            nn.ReLU(),
            nn.BatchNorm1d(channel_outs // 4)
        )

        # 尺寸编码 MLP：6 (sin&cos) -> 64
        self.size_mlp = nn.Sequential(
            nn.Linear(6, channel_outs // 4),
            nn.ReLU(),
            nn.BatchNorm1d(channel_outs // 4)
        )

        # 语义特征 MLP：6 (sin&cos) -> 64
        self.sem_mlp = nn.Sequential(
            nn.Linear(channel_outs*2, channel_outs // 4),
            nn.ReLU(),
            nn.BatchNorm1d(channel_outs // 4)
        )

        self.aux_fusion_mlp = FCResLayer(channel_outs//4*3)

        self.fusion_mlp = nn.Sequential(
            FCResLayer(channel_outs),
            FCResLayer(channel_outs),
            FCResLayer(channel_outs),
        )

    def forward(self, meta_inf: list, x_dtype: torch.dtype, device: torch.device, sem_inf: torch.Tensor) -> torch.Tensor:
        """处理元数据并融合补偿特征，生成最终辅助特征

        Args:
            meta_inf (list): 包含样本元数据的列表，每个元素需有 view/width/height/band_type 属性
            x_dtype (torch.dtype): 目标数据类型
            device (torch.device): 目标设备
            sem_inf (torch.Tensor): 语义补偿特征

        Returns:
            torch.Tensor: 维度为 [batch_size, channel_outs] 的融合特征
        """
        # 视角的One-Hot编码 (3维)
        view_one_hot = torch.stack([
            torch.tensor([1 if sample.view == category else 0
                          for category in ["Air", "Space", "Land"]],
                         device=device, dtype=x_dtype)
            for sample in meta_inf
        ])  # [B,3]

        # 波段的One-Hot编码 (3维)
        band_one_hot = torch.stack([
            torch.tensor([1 if sample.band_type == category else 0
                          for category in ["LWIR", "NIR", "SWIR"]],
                         device=device, dtype=x_dtype)
            for sample in meta_inf
        ])  # [B,3]

        # 尺寸特征计算
        w = torch.stack([torch.tensor(sample.width, device=device, dtype=x_dtype)
                         for sample in meta_inf])  # [B]
        h = torch.stack([torch.tensor(sample.height, device=device, dtype=x_dtype)
                         for sample in meta_inf])  # [B]
        size_features = torch.stack([
            w / 2048.0,
            h / 2048.0,
            w / (h + 1e-6)
        ], dim=-1)  # [B,3]
        size_transformed = torch.cat([
            torch.sin(size_features),
            torch.cos(size_features)
        ], dim=-1)  # [B,6]

        # 通过 MLP 提升维度
        # 元数据特征 均提升至 [B, channel_outs//4]
        view_feat = self.view_mlp(view_one_hot)
        band_feat = self.band_mlp(band_one_hot)
        size_feat = self.size_mlp(size_transformed)

        # 补偿特征 也提升至 [B, channel_outs//4]
        sem_feat = self.sem_mlp(sem_inf)

        # 元数据特征融合 [B, channel_outs//4*3]
        vsb_features = self.aux_fusion_mlp(torch.cat([view_feat, band_feat, size_feat], dim=-1))  # [B,256]

        # 元数据特征-补偿特征融合 [B, channel_outs]
        all_aux_inf = self.fusion_mlp(torch.cat([vsb_features, sem_feat], dim=-1))

        return all_aux_inf


class EdgeConvSep(nn.Module):
    """分解式边缘卷积层 (3x1 + 1x3) 稳定版"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 垂直方向卷积 (3x1)
        self.conv_vert = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=(3, 1),
                                   padding=(1, 0),
                                   bias=False)
        # 水平方向卷积 (1x3)
        self.conv_hori = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=(1, 3),
                                   padding=(0, 1),
                                   bias=False)

        # 稳定化参数
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始小量
        self.beta = nn.Parameter(torch.ones(1))  # 缩放因子
        self.register_buffer('epsilon', torch.tensor(1e-5))  # 稳定常数

        # 精确初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """稳定的对称初始化"""
        with torch.no_grad():
            # 垂直核初始化（Sobel垂直核变体）
            vert_kernel = torch.tensor([[[[1.], [-2.], [1.]]]]).repeat(
                self.conv_vert.out_channels, self.conv_vert.in_channels, 1, 1)
            self.conv_vert.weight.copy_(vert_kernel / 4.0)  # 缩小初始值

            # 水平核初始化（Sobel水平核变体）
            hori_kernel = torch.tensor([[[[1., -2., 1.]]]]).repeat(
                self.conv_hori.out_channels, self.conv_hori.in_channels, 1, 1)
            self.conv_hori.weight.copy_(hori_kernel / 4.0)

    def _stable_combination(self, x: Tensor) -> Tensor:
        """带稳定化处理的卷积组合"""
        # 垂直卷积 + 激活函数约束
        x = F.relu(self.conv_vert(x)) * torch.sigmoid(self.alpha)

        # 水平卷积 + 归一化
        x = self.conv_hori(x)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + self.epsilon
        return (x - mean) / std * self.beta

    def forward(self, x: Tensor) -> Tensor:
        # 稳定的前向计算
        return self._stable_combination(x)


class SpatialModulation(nn.Module):
    """逐样本空间调制模块"""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def _parse_conv_params(self, spatial_att_weights):
        B = spatial_att_weights.size(0)
        params = {}

        # 参数切片 (2通道×4参数=8)
        conv_params = spatial_att_weights[:, :8].view(B, 2, 4)  # [B,2,4]

        # 初始化3x3卷积核 (四角置零)
        conv_kernels = torch.zeros(B, 2, 3, 3, device=spatial_att_weights.device)

        # 填充上下左右四个位置 (中间行/列)
        # 上 (row=0, col=1)
        conv_kernels[:, :, 0, 1] = conv_params[:, :, 0]
        # 下 (row=2, col=1)
        conv_kernels[:, :, 2, 1] = conv_params[:, :, 1]
        # 左 (row=1, col=0)
        conv_kernels[:, :, 1, 0] = conv_params[:, :, 2]
        # 右 (row=1, col=2)
        conv_kernels[:, :, 1, 2] = conv_params[:, :, 3]

        # 计算中心值 (周围四个值的负和)
        center = -(conv_params.sum(dim=2))  # sum(4个参数)的负数 [B,2]
        conv_kernels[:, :, 1, 1] = center

        params['conv3_weight'] = conv_kernels.unsqueeze(1)  # [B,1,2,3,3]
        return params

    def forward(self, x, spatial_att_weights):
        B, C, H, W = x.shape

        # Step1: 生成通道特征 [B,2,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)

        # Step2: 解析参数
        with torch.no_grad():
            params = self._parse_conv_params(spatial_att_weights)

        # Step3: 重构输入特征为分组卷积格式
        y_group = y.view(1, B * 2, H, W)  # [1, B*2, H, W]

        # Step4: 分组卷积操作
        def _conv_ops(y_group, weight, kernel_size):
            # 调整权重维度 [B,1,2,3,3] → [B*1,2,3,3]
            weight = weight.view(B * 1, 2, kernel_size, kernel_size)
            # 执行分组卷积 (无偏置)
            return F.conv2d(
                y_group,
                weight=weight.detach(),
                bias=None,
                padding=kernel_size // 2,
                stride=1,
                groups=B
            )

        # 3x3卷积
        y3 = _conv_ops(y_group, params['conv3_weight'], 3)
        y3 = y3.view(B, 1, H, W)  # [B,1,H,W]

        # Step5: 应用注意力
        return x * self.sigmoid(y3) + x