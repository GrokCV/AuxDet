import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class DMLPAttention(nn.Module):
    """动态MLP注意力模块

        该模块通过动态 MLP实现特征交互并生成注意力权重，用于动态特征调制。

        参数:
            in_channels_f1 (int): 输入特征f1的通道数，默认为256
        """
    def __init__(self, in_channels_f1=256, ):
        super().__init__()
        # 输入特征通道数
        self.in_channels_f1 = in_channels_f1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 动态MLP层，用于生成注意力权重
        # 输入: 降维后的 f1 的图像特征 + 辅助特征(vs_features)
        # 输出: 注意力权重(通道注意力+空间注意力)
        # 参数说明:
        #   in_channels_f1: 输入特征维度
        #   in_channels_f1//2: 中间层维度
        #   in_channels_f1+8: 输出维度(通道注意力256 + 空间注意力8)
        #   32: 动态MLP的隐藏层维度
        #   2: 动态MLP的层数
        #   'a': 动态MLP的类型标识
        self.loc_att = get_dynamic_mlp(in_channels_f1, in_channels_f1//2, in_channels_f1+8, 32, 2, 'a') #169  256+  +8


    def forward(self, f1, vs_features):
        f1_1 = self.avgpool(f1)
        f1_1 = torch.flatten(f1_1, 1)

        f1_att = self.loc_att(f1_1, vs_features)

        return f1_att


@MODELS.register_module()
class FCResLayer(nn.Module):
    def __init__(self, linear_size=256):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes=1000, num_filts=256, return_feats=False):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, num_filts),
            nn.ReLU(inplace=True),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
        )
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.return_feats = return_feats

    def forward(self, x, return_feats=False, class_of_interest=None):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb  # [b, num_filts]
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return class_pred  # [b, num_class]

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(conv, )
        if not bias:
            self.conv.add_module('ln', nn.LayerNorm(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Dynamic_MLP_A(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.get_weight = nn.Linear(loc_planes, inplanes * planes)
        self.norm = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        weight = self.get_weight(loc_fea)
        weight = weight.view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(img_fea.unsqueeze(1), weight).squeeze(1)
        img_fea = self.norm(img_fea)
        img_fea = self.relu(img_fea)

        return img_fea


class Dynamic_MLP_B(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        weight11 = self.conv11(img_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(loc_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea



class Dynamic_MLP_C(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )

        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        cat_fea = torch.cat([img_fea, loc_fea], 1)

        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(cat_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)

        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea


class RecursiveBlock(nn.Module):
    def __init__(self, inplanes, planes, loc_planes, mlp_type='c'):
        super().__init__()
        if mlp_type.lower() == 'a':
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == 'b':
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == 'c':
            MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def forward(self, img_fea, loc_fea):
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea


class FusionModule(nn.Module):
    def __init__(self, inplanes=2048, planes=256, outplanes=64*32*32, hidden=64, num_layers=2, mlp_type='c'):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)

        # self.conv3 = nn.Linear(planes, outplanes)
        # self.norm3 = nn.LayerNorm(outplanes)
        # self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Linear(planes, inplanes)
        self.norm3 = nn.LayerNorm(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Linear(inplanes, outplanes)
        self.norm4 = nn.LayerNorm(outplanes)

    def forward(self, img_fea, loc_fea):
        '''
        img_fea: (N, channel), backbone输出经过全局池化的feature
        loc_fea: (N, fea_dim)
        '''
        identity = img_fea

        img_fea = self.conv1(img_fea)

        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        # return img_fea

        img_fea += identity

        return self.norm4(self.conv4(img_fea))


def get_dynamic_mlp(inplanes, mlp_d, outplanes, mlp_h, mlp_n, mlp_type):
    return FusionModule(inplanes=inplanes,
                        planes=mlp_d,
                        outplanes=outplanes,
                        hidden=mlp_h,
                        num_layers=mlp_n,
                        mlp_type=mlp_type,
                        )







