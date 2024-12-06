import torch
import torch.nn as nn
import timm

from models.SEAtten import seC3
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

class MultiScaleFeature(nn.Module):
    def __init__(self, out_indices=[0, 1, 2, 3], fpn_out_channels=256): # , fpn_out_channels=256
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

class pre_trained_model6(nn.Module):

    # 把num_classes=1改成了=4 num_classes当成预测速度
    # distype作为失真类别的数量
    def __init__(self, img_size=14,
                 patch_size=1,
                 in_chans=1024,
                 num_classes=4,
                 distype=5,
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
                 pos_econding=False,
                 con3d=False,
                 block_frame=8,
                 out_indices=3,
                 head_dim=1024,
                 block_num=4,
                 activate_C=False,
                 activate_N=False,
                 imgSize=224,
                 patchSize=8):

        super().__init__()
        self.num_classes = num_classes
        self.distype = distype
        self.con3d = con3d
        self.conv3d = nn.Conv3d(1, 3, kernel_size=(block_frame, 3, 3), stride=(block_frame, 1, 1), padding=(0, 1, 1))
        self.embed_dim = embed_dim
        self.patches_resolution = (imgSize // patchSize, imgSize // patchSize)

        self.multiscale_feature = MultiScaleFeature(out_indices=[0, 1, 2, 3])

        self.act = nn.ReLU(inplace=True)

        # TODO: 3D-pool
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.se = seC3(head_dim, head_dim, n=1, shortcut=True, g=1, e=0.5)


        self.tempconv = nn.Conv2d(1024, 256, 1, 1, 0)

        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=[2, 2],
            num_heads=[2, 4],
            embed_dim=256,
            window_size=4,
            dim_mlp=256,
            scale=0.8
        )


        self.linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(drop_rate)
        )


        #  新增mlp层
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        self.pace = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(128, self.num_classes)
        )
        self.distype = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(128, self.distype)
        )

    def forward(self, rgb_clip, source_clip=None):
        if source_clip is not None:
            if self.con3d:
                rgb_clip = self.conv3d(rgb_clip)
                source_clip = self.conv3d(source_clip)
                # x = self.act(x)
            # print(f"after con3d : {rgb_clip.shape}")
            B, C, D, H, W = rgb_clip.size()  # batch=8 C=3 D=16 H=112 W=112
            # B D C H W
            rgb_clip = rgb_clip.permute(0, 2, 1, 3, 4)  # rgb_clip Tensor(8, 16, 3, 112, 112)
            source_clip = source_clip.permute(0, 2, 1, 3, 4)
            # B*D C H W
            rgb_clip = rgb_clip.contiguous().view(B * D, C, H, W)  # Tensor(128, 3, 112, 112)
            source_clip = source_clip.contiguous().view(B * D, C, H, W)

            # TODO : CNN-model
            # rgb_clip = self.backbone(rgb_clip)[0]
            # source_clip = self.backbone(source_clip)[0]
            rgb_clip = self.multiscale_feature(rgb_clip)  # Tensor(128, 2048, 1, 1)
            source_clip = self.multiscale_feature(source_clip)
            # rgb_clip = self.conv_tmp(rgb_clip)

            rgb_clip = self.se(rgb_clip)
            source_clip = self.se(source_clip)

            rgb_clip = self.tempconv(rgb_clip)
            rgb_clip = self.swintransformer1(rgb_clip)
            source_clip = self.tempconv(source_clip)
            source_clip = self.swintransformer1(source_clip)

            # TODO: Transformer
            rgb_clip = self.pool(rgb_clip)
            source_clip = self.pool(source_clip)
            #
            _C, _H, _W = rgb_clip.shape[-3], rgb_clip.shape[-2], rgb_clip.shape[-1]
            rgb_clip = rgb_clip.contiguous().view(B, D, _C)  # (8, 16, 2048)
            source_clip = source_clip.contiguous().view(B, D, _C)

            rgb_clip = self.linear(rgb_clip)  # (8, 16, 256)
            source_clip = self.linear(source_clip)

            # print(f"batch : {B}, depth : {D}, channel : {}")

            # TODO : transformer
            # rgb_clip = self.transformer(rgb_clip)  # tensor(5, 16, 256)
            # source_clip = self.transformer(source_clip)  # tensor(5, 16, 256)

            rgb_clip = rgb_clip.mean(dim=1)  # tensor(5, 256)
            source_clip = source_clip.mean(dim=1)  # tensor(5, 256)

            # 新加mlp
            # c_rgb_clip = self.projection_head(rgb_clip)
            # TODO: END
            pace = self.pace(source_clip)
            distype = self.distype(rgb_clip - source_clip)
            # 多加入返回特征rgb_clip，特征用于对比学习中
            # return pace, distype, rgb_clip
            return pace, distype

        else:
            if self.con3d:
                rgb_clip = self.conv3d(rgb_clip)
                # x = self.act(x)
            # # print(f"after con3d : {rgb_clip.shape}")
            B, C, D, H, W = rgb_clip.size()
            # # B D C H W
            rgb_clip = rgb_clip.permute(0, 2, 1, 3, 4)
            # # B*D C H W
            rgb_clip = rgb_clip.contiguous().view(B * D, C, H, W)

            # TODO : CNN-model
            rgb_clip = self.multiscale_feature(rgb_clip)
            # rgb_clip = self.conv_tmp(rgb_clip)

            rgb_clip = self.se(rgb_clip)

            rgb_clip = self.tempconv(rgb_clip)
            rgb_clip = self.swintransformer1(rgb_clip)

            # TODO: Transformer
            rgb_clip = self.pool(rgb_clip)

            _C, _H, _W = rgb_clip.shape[-3], rgb_clip.shape[-2], rgb_clip.shape[-1]
            rgb_clip = rgb_clip.contiguous().view(B, D, _C)

            rgb_clip = self.linear(rgb_clip)
            # print(f"batch : {B}, depth : {D}, channel : {}")

            # TODO : transformer
            # rgb_clip = self.transformer(rgb_clip)  # tensor(5, 16, 256)
            rgb_clip = rgb_clip.mean(dim=1)  # tensor(5, 256)

            # mlp新增
            c_rgb_clip = self.projection_head(rgb_clip)
            # TODO: END
            pace = self.pace(rgb_clip)

            # 多加入返回特征rgb_clip，特征用于对比学习中
            # return pace, dislevel, rgb_clip
            return pace, c_rgb_clip
            # return pace, rgb_clip



