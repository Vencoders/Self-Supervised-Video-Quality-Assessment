import torch
import torch.nn as nn
import timm
import sys

sys.path.append('/data/wst/video-pace')

from models.simAM import SimAM

from models.swin import SwinTransformer


class FPN(nn.Module):
    def __init__(self, in_channels, in_channels2, in_channels3, in_channels4, out_channels):
        super(FPN, self).__init__()
        self.p6_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.p5_conv = nn.Conv2d(in_channels2, out_channels, kernel_size=1, stride=1, padding=0)
        self.p4_conv = nn.Conv2d(in_channels3, out_channels, kernel_size=1, stride=1, padding=0)
        self.p3_conv = nn.Conv2d(in_channels4, out_channels, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()

    def forward(self, inputs):
        c3, c4, c5, c6 = inputs
        p6 = self.p6_conv(c6)
        p5 = self.p5_conv(c5) + self.upsample(p6)
        p4 = self.p4_conv(c4) + self.upsample(p5)
        p3 = self.p3_conv(c3) + self.upsample(p4)
        return [p3, p4, p5, p6]

# class AdaFeatBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.kernel_field = kernel_size * kernel_size
#         self.offset_mask_conv = nn.Conv2d(
#             in_channels=in_channel, out_channels=self.kernel_field*3,
#             kernel_size=kernel_size, stride=stride, padding=padding
#         )
#         # nn.init.constant_(self.offset_mask_conv.weight, 0.)
#         # nn.init.constant_(self.offset_mask_conv.bias, 0.)
#         self.deform_conv = ModulatedDeformConv2d(in_channel, out_channel, kernel_size, stride, padding)
#         # self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         # self.conv = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
#
#     def forward(self, x):
#         offset, mask = torch.split(
#             self.offset_mask_conv(x),
#             [self.kernel_field*2, self.kernel_field], dim=1
#         )
#         out = self.deform_conv(
#             x, offset.contiguous(),
#             2*torch.sigmoid(mask.contiguous())
#         )
#         # return self.conv(self.relu(out)) + x
#         return out
#
#
# class FPN(nn.Module):
#     def __init__(self, in_channels, in_channels2, in_channels3, in_channels4, out_channels):
#         super(FPN, self).__init__()
#         # self.p6_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         # self.p5_conv = nn.Conv2d(in_channels2, out_channels, kernel_size=1, stride=1, padding=0)
#         # self.p4_conv = nn.Conv2d(in_channels3, out_channels, kernel_size=1, stride=1, padding=0)
#         # self.p3_conv = nn.Conv2d(in_channels4, out_channels, kernel_size=1, stride=1, padding=0)
#
#         # self.p6_conv = AdaFeatBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         # self.p5_conv = AdaFeatBlock(in_channels2, out_channels, kernel_size=3, stride=1, padding=1)
#         # self.p4_conv = AdaFeatBlock(in_channels3, out_channels, kernel_size=3, stride=1, padding=1)
#         # self.p3_conv = AdaFeatBlock(in_channels4, out_channels, kernel_size=3, stride=1, padding=1)
#
#         self.p6_conv = AdaFeatBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.p5_conv = AdaFeatBlock(in_channels2, out_channels, kernel_size=3, stride=1, padding=1)
#         # self.p4_conv = AdaFeatBlock(in_channels3, out_channels, kernel_size=3, stride=1, padding=1)
#         # self.p3_conv = AdaFeatBlock(in_channels4, out_channels, kernel_size=3, stride=1, padding=1)
#         self.p4_conv = nn.Conv2d(in_channels3, out_channels, kernel_size=3, stride=1, padding=1)
#         self.p3_conv = nn.Conv2d(in_channels4, out_channels, kernel_size=3, stride=1, padding=1)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.relu = nn.ReLU()
#
#     def forward(self, inputs):
#         c3, c4, c5, c6 = inputs
#         p6 = self.p6_conv(c6)
#         p5 = self.p5_conv(c5) + self.upsample(p6)
#         p4 = self.p4_conv(c4) + self.upsample(p5)
#         p3 = self.p3_conv(c3) + self.upsample(p4)
#         return [p3, p4, p5, p6]

class MultiScaleFeature(nn.Module):
    def __init__(self, out_indices=[0, 1, 2, 3], fpn_out_channels=64, att_out_channels=1024): # , fpn_out_channels=256---64---128
        super(MultiScaleFeature, self).__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=out_indices)
        self.fpn = FPN(self.backbone.feature_info[out_indices[-1]]['num_chs'], self.backbone.feature_info[out_indices[-2]]['num_chs'], self.backbone.feature_info[out_indices[-3]]['num_chs'], self.backbone.feature_info[out_indices[-4]]['num_chs'], out_channels=fpn_out_channels)  # 将out_channels参数作为关键字参数传递给FPN模块  ,out_channels=fpn_out_channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))

    def forward(self, x):
        feats = []
        outs = self.backbone(x)
        outs = outs[-4:]
        p3, p4, p5, p6 = self.fpn(outs)
        out = self.adaptive_pool(p6)
        feats.append(out)
        out = self.adaptive_pool(p5)
        feats.append(out)
        out = self.adaptive_pool(p4)
        feats.append(out)
        out = self.adaptive_pool(p3)
        feats.append(out)
        out = torch.cat(feats, dim=1)
        return out

class VB_CNN_Transformer2(nn.Module):
    def __init__(self, img_size=14,
                 patch_size=1,
                 in_chans=1024,
                 num_classes=1,
                 embed_dim=256,
                 depths=[2, 6],
                 num_heads=[4, 16],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 ape=False,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 attn_drop_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pos_econding=True,
                 con3d=False,
                 block_frame=8, # 8改成4
                 out_indices=3,
                 head_dim=1024,
                 num_outputs=1,
                 imgSize=224,
                 patchSize=8):

        super().__init__()
        self.con3d = con3d
        self.conv3d = nn.Conv3d(1, 3, kernel_size=(
            block_frame, 3, 3), stride=(block_frame, 1, 1), padding=(0, 1, 1))
        self.embed_dim = embed_dim
        self.patches_resolution = (imgSize // patchSize, imgSize // patchSize)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))

        # backbone消融实验
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.multiscale_feature = MultiScaleFeature(out_indices=[0, 1, 2, 3])

        # self.backbone = timm.create_model(
        #     'tf_efficientnetv2_s', pretrained=True, features_only=True, out_indices=[out_indices])

        # self.backbone = timm.create_model('vgg16', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('densenet161', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('resnext50_32x4d', pretrained=True, features_only=True, out_indices=[out_indices])

        # self.conv_tmp = nn.Sequential(
        #     nn.Conv2d(160, 1024, 3, 1, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )
        self.conv_tmp = nn.Conv2d(1024, 256, 1, 1, 0)  # 1024-->256-->512-->1024  256-->128-->256

        # self.act = self.backbone.act1 if 'act1' in self.backbone else nn.ReLU(inplace=True)
        self.act = nn.ReLU(inplace=True)

        self.pos_econding = pos_econding
        if self.pos_econding:
            self.PosNet = PosNet(in_plane=64 + 256 + 512 +
                                 1024, out_plane=embed_dim, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # TODO: 3D-pool
        # self.pool = nn.AdaptiveAvgPool3d(1)
        # self.se = seC3(256, 256, n=1, shortcut=True, g=1, e=0.5)
        # self.se = seC3(1024, head_dim, n=1, shortcut=True, g=1, e=0.5)
        # self.ema = EMA(head_dim, factor=32)
        # self.gam =GAM_Attention(head_dim, head_dim)
        self.simam = SimAM(head_dim, head_dim)
        # self.simam = SimAM(512, 512)

        # self.se = seC3(512, 512, n=1, shortcut=True, g=1, e=0.5)  # head_dim-->256-->512
        # self.cbam = Conv_CBAM(1024, 1024, k=1, s=1, p=None, g=1, act=True)
        # self.eca = Eca_layer(1024, k_size=3)


        # self.transformer = Transformer(dim=256,
        #                                depth=4,
        #                                heads=8,
        #                                dim_head=128,
        #                                mlp_dim=head_dim,
        #                                dropout=drop_rate)
        # self.tempconv = nn.Conv2d(1024, 256, 1, 1, 0)
        # self.tempconv = nn.Conv2d(512, 256, 1, 1, 0)  # 1024-->256-->512 256-->128-->256
        # self.tempconv = nn.Conv2d(256, 128, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=[2, 2],
            num_heads=[2, 2, 4],
            embed_dim=256,
            window_size=4,
            dim_mlp=256,
            scale=0.8
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.Dropout(drop_rate)
        # )
        self.linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(drop_rate)
        )

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        if self.con3d:
            x = self.conv3d(x)
        # print(f"after con3d : {x.shape}")
        B, C, D, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # B D C H W
        x = x.permute(0, 2, 1, 3, 4)
        # B*D C H W
        x = x.contiguous().view(B * D, C, H, W)

        # TODO : CNN-model
        x = self.backbone(x)[0]
        x = self.adaptive_pool(x)

        # x = self.multiscale_feature(x)  # (64, 2048, 1, 1)

        # x = self.conv_tmp(x)
        # x = self.cbam(x)
        # x = self.eca(x)
        # x = self.se(x)
        # x = self.ema(x)
        # x = self.gam(x)
        x = self.simam(x)
        x = self.conv_tmp(x)

        # x = self.tempconv(x)
        x = self.swintransformer1(x)

        # 消融实验pool和transformer
        # TODO: 3D-pool
        # _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
        # x = x.contiguous().view(B, D, _C, _H, _W).permute(0, 2, 1, 3, 4)
        # x = self.pool(x).view(B, _C)
        # x = self.linear(x)

        # x = self.se(x)
        # TODO: Transformer
        x = self.pool(x)
        _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.contiguous().view(B, D, _C)
        x = self.linear(x) # (16, 4, 256)
        # x = self.transformer(x)
        x = x.mean(dim=1) #(16, 256)

        x = self.head(x)

        return x


# 简单resnet 两个拼接
# class MultiScaleFeature(nn.Module):
#     def __init__(self, out_indices=[0, 1, 2]):
#         super(MultiScaleFeature, self).__init__()
#         self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=out_indices)
#         self.out_indices = out_indices
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.conv = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)  # 添加一个额外的卷积层
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     def forward(self, x):
#         feats = []
#         for size in [256, 512]:
#             out = self.backbone.conv1(x)
#             out = self.backbone.bn1(out)
#             out = self.backbone.act1(out)
#             out = self.backbone.maxpool(out)
#             out = self.backbone.layer1(out)
#             out = self.backbone.layer2(out)
#             out = self.backbone.layer3(out)
#             out = self.maxpool(out)
#             out = self.adaptive_pool(out)
#             feats.append(out)
#             x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
#         out = torch.cat(feats, dim=1)
#         return out

# FPN形式实现多尺度 三层拼接
# class FPN(nn.Module):
#     def __init__(self, in_channels, in_channels2, in_channels3, out_channels):
#         super(FPN, self).__init__()
#         self.p5_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.p4_conv = nn.Conv2d(in_channels2, out_channels, kernel_size=1, stride=1, padding=0)
#         self.p3_conv = nn.Conv2d(in_channels3, out_channels, kernel_size=1, stride=1, padding=0)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.relu = nn.ReLU()
#
#     def forward(self, inputs):
#         c3, c4, c5 = inputs
#         p5 = self.p5_conv(c5)
#         p4 = self.p4_conv(c4) + self.upsample(p5)
#         p3 = self.p3_conv(c3) + self.upsample(p4)
#         return [p3, p4, p5]
#
# class MultiScaleFeature(nn.Module):
#     def __init__(self, out_indices=[0, 1, 2], fpn_out_channels=256): # , fpn_out_channels=256
#         super(MultiScaleFeature, self).__init__()
#         self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=out_indices)
#         self.fpn = FPN(self.backbone.feature_info[out_indices[-1]]['num_chs'], self.backbone.feature_info[out_indices[-2]]['num_chs'], self.backbone.feature_info[out_indices[-3]]['num_chs'], out_channels=fpn_out_channels)  # 将out_channels参数作为关键字参数传递给FPN模块  ,out_channels=fpn_out_channels
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         feats = []
#         outs = self.backbone(x)
#         outs = outs[-3:]
#         p3, p4, p5 = self.fpn(outs)
#         out = self.adaptive_pool(p5)
#         feats.append(out)
#         out = self.adaptive_pool(p4)
#         feats.append(out)
#         out = self.adaptive_pool(p3)
#         feats.append(out)
#         out = torch.cat(feats, dim=1)
#         return out

# class FPN(nn.Module):
#     def __init__(self, in_channels, in_channels2, in_channels3, in_channels4, out_channels):
#         super(FPN, self).__init__()
#         self.p6_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.p5_conv = nn.Conv2d(in_channels2, out_channels, kernel_size=1, stride=1, padding=0)
#         self.p4_conv = nn.Conv2d(in_channels3, out_channels, kernel_size=1, stride=1, padding=0)
#         self.p3_conv = nn.Conv2d(in_channels4, out_channels, kernel_size=1, stride=1, padding=0)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.relu = nn.ReLU()
#
#     def forward(self, inputs):
#         c3, c4, c5, c6 = inputs
#         p6 = self.p6_conv(c6)
#         p5 = self.p5_conv(c5) + self.upsample(p6)
#         p4 = self.p4_conv(c4) + self.upsample(p5)
#         p3 = self.p3_conv(c3) + self.upsample(p4)
#         return [p3, p4, p5, p6]
#
# class MultiScaleFeature(nn.Module):
#     def __init__(self, out_indices=[0, 1, 2, 3], fpn_out_channels=256, att_out_channels=1024): # , fpn_out_channels=256
#         super(MultiScaleFeature, self).__init__()
#         self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=out_indices)
#         self.fpn = FPN(self.backbone.feature_info[out_indices[-1]]['num_chs'], self.backbone.feature_info[out_indices[-2]]['num_chs'], self.backbone.feature_info[out_indices[-3]]['num_chs'], self.backbone.feature_info[out_indices[-4]]['num_chs'], out_channels=fpn_out_channels)  # 将out_channels参数作为关键字参数传递给FPN模块  ,out_channels=fpn_out_channels
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#
#     def forward(self, x):
#         feats = []
#         outs = self.backbone(x)
#         outs = outs[-4:]
#         p3, p4, p5, p6 = self.fpn(outs)
#         out = self.adaptive_pool(p6)
#         feats.append(out)
#         out = self.adaptive_pool(p5)
#         feats.append(out)
#         out = self.adaptive_pool(p4)
#         feats.append(out)
#         out = self.adaptive_pool(p3)
#         feats.append(out)
#         out = torch.cat(feats, dim=1)
#         return out
#
# # class MultiScaleFeature(nn.Module):
# #     def __init__(self, out_indices=[0, 1, 2, 3]):
# #         super(MultiScaleFeature, self).__init__()
# #         self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=out_indices)
# #         self.out_indices = out_indices
# #         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
# #
# #     def forward(self, x):
# #         feats = []
# #         for size in [256, 512]:
# #             out = self.backbone(x) # x(128, 3, 1024, 1024),
# #             out = self.adaptive_pool(out[self.out_indices[-1]]) # out(128, 1024, 1, 1)
# #             feats.append(out)
# #
# #             x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
# #
# #         out = torch.cat(feats, dim=1) # out(128,2048,1,1)
# #         # out = out.view(out.size(0), -1)
# #         return out
#
# # class MultiScaleFeature(nn.Module):
# #     def __init__(self, out_indices=[0, 1, 2, 3]):
# #         super(MultiScaleFeature, self).__init__()
# #
# #         # 加载预训练的 ViT 模型
# #         self.backbone = timm.create_model('vit_base_patch8_224', pretrained=True)
# #
# #         # 保存需要输出的层
# #         self.out_indices = out_indices
# #
# #         # 定义自适应池化层
# #         self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
# #
# #     def forward(self, x):
# #         feats = []
# #
# #         # 对输入图像进行多尺度下采样和特征池化
# #         for size in [256, 512]:
# #             # 将输入图像分成多个 patch，并将它们展平成一维序列
# #             patch_size = size // 16  # patch size = 16
# #             x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous()
# #             x = x.view(-1, 3, patch_size, patch_size)
# #             x = x.permute(0, 2, 3, 1)  # 将通道维度放到最后
# #
# #             # 使用 ViT 模型提取特征
# #             out = self.backbone(x)
# #
# #             # 将特征张量展平，并进行自适应池化
# #             out = out.view(out.size(0), out.size(1), -1)
# #             out = self.adaptive_pool(out).squeeze(-1)
# #
# #             # 将特征添加到列表中
# #             feats.append(out)
# #
# #             # 对输入图像进行下采样
# #             x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
# #
# #         # 在通道维度上拼接所有尺度下的特征表示
# #         out = torch.cat(feats, dim=1)
# #
# #         return out
#
# class VB_CNN_Transformer2(nn.Module):
#     def __init__(self, img_size=14,
#                  patch_size=1,
#                  in_chans=1024,
#                  num_classes=1,
#                  embed_dim=256,
#                  depths=[2, 6],
#                  num_heads=[4, 16],
#                  window_size=7,
#                  mlp_ratio=4.0,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  ape=False,
#                  drop_rate=0.1,
#                  drop_path_rate=0.1,
#                  attn_drop_rate=0.1,
#                  patch_norm=True,
#                  use_checkpoint=False,
#                  pos_econding=True,
#                  con3d=False,
#                  block_frame=8, # 8改成4
#                  out_indices=3,
#                  head_dim=1024,
#                  num_outputs=1):
#
#         super().__init__()
#         self.con3d = con3d
#         self.conv3d = nn.Conv3d(1, 3, kernel_size=(
#             block_frame, 3, 3), stride=(block_frame, 1, 1), padding=(0, 1, 1))
#         self.embed_dim = embed_dim
#
#         # backbone消融实验
#         # self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=[out_indices])
#         self.multiscale_feature = MultiScaleFeature(out_indices=[0, 1, 2, 3])
#
#         # self.backbone = timm.create_model(
#         #     'tf_efficientnetv2_s', pretrained=True, features_only=True, out_indices=[out_indices])
#
#         # self.backbone = timm.create_model('vgg16', pretrained=True, features_only=True, out_indices=[out_indices])
#         # self.backbone = timm.create_model('densenet161', pretrained=True, features_only=True, out_indices=[out_indices])
#         # self.backbone = timm.create_model('resnext50_32x4d', pretrained=True, features_only=True, out_indices=[out_indices])
#
#         # self.conv_tmp = nn.Sequential(
#         #     nn.Conv2d(160, 1024, 3, 1, 1),
#         #     nn.BatchNorm2d(1024),
#         #     nn.ReLU(inplace=True)
#         # )
#
#         # self.act = self.backbone.act1 if 'act1' in self.backbone else nn.ReLU(inplace=True)
#         self.act = nn.ReLU(inplace=True)
#
#         self.pos_econding = pos_econding
#         if self.pos_econding:
#             self.PosNet = PosNet(in_plane=64 + 256 + 512 +
#                                  1024, out_plane=embed_dim, kernel_size=3)
#
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         # TODO: 3D-pool
#         # self.pool = nn.AdaptiveAvgPool3d(1)
#
#         # self.se = seC3(1024, head_dim, n=1, shortcut=True, g=1, e=0.5)
#         # self.cbam = Conv_CBAM(1024, 1024, k=1, s=1, p=None, g=1, act=True)
#         self.eca = Eca_layer(1024, k_size=3)
#
#
#         self.transformer = Transformer(dim=256,
#                                        depth=4,
#                                        heads=8,
#                                        dim_head=128,
#                                        mlp_dim=head_dim,
#                                        dropout=drop_rate)
#
#         # self.linear = nn.Sequential(
#         #     nn.Linear(768, 256),
#         #     nn.Dropout(drop_rate)
#         # )
#         self.linear = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.Dropout(drop_rate)
#         )
#
#         self.head = nn.Sequential(
#             nn.Dropout(drop_rate),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#         if self.con3d:
#             x = self.conv3d(x)
#         # print(f"after con3d : {x.shape}")
#         B, C, D, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
#         # B D C H W
#         x = x.permute(0, 2, 1, 3, 4)
#         # B*D C H W
#         x = x.contiguous().view(B * D, C, H, W)
#
#         # TODO : CNN-model
#         # x = self.backbone(x)[0]
#         x = self.multiscale_feature(x)  # (64, 2048, 1, 1)
#
#         # x = self.conv_tmp(x)
#         # x = self.cbam(x)
#         x = self.eca(x)
#
#         # 消融实验pool和transformer
#         # TODO: 3D-pool
#         # _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
#         # x = x.contiguous().view(B, D, _C, _H, _W).permute(0, 2, 1, 3, 4)
#         # x = self.pool(x).view(B, _C)
#         # x = self.linear(x)
#
#         # x = self.se(x)
#         # TODO: Transformer
#         x = self.pool(x)
#         _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
#         x = x.contiguous().view(B, D, _C)
#         x = self.linear(x) # (16, 4, 256)
#         x = self.transformer(x)
#         x = x.mean(dim=1) #(16, 256)
#
#         x = self.head(x)
#
#         return x



