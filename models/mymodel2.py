import torch.nn as nn
import timm
# from .vit import Transformer
from models.vit import Transformer
# from vit import Transformer
import torch
from models.SEAtten import seC3


class MultiScaleFeature(nn.Module):
    def __init__(self, out_indices=[0, 1, 2]):
        super(MultiScaleFeature, self).__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=out_indices)
        self.out_indices = out_indices
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)  # 添加一个额外的卷积层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feats = []
        for size in [256, 512]:
            out = self.backbone.conv1(x)
            out = self.backbone.bn1(out)
            out = self.backbone.act1(out)
            out = self.backbone.maxpool(out)

            out = self.backbone.layer1(out)
            out = self.backbone.layer2(out)
            out = self.backbone.layer3(out)
            out = self.maxpool(out)
            out = self.adaptive_pool(out)
            feats.append(out)

            x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        out = torch.cat(feats, dim=1)
        return out

class pre_trained_model2(nn.Module):

    # 把num_classes=1改成了=4 num_classes当成预测速度
    # distype作为失真类别的数量
    def __init__(self, img_size=14,
                 patch_size=1,
                 in_chans=1024,
                 num_classes=4,
                 distype=5,
                 dis_cd=3,
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
                 block_num=4):

        super().__init__()
        self.num_classes = num_classes
        self.distype = distype
        self.dis_cd = dis_cd
        self.con3d = con3d
        self.conv3d = nn.Conv3d(1, 3, kernel_size=(block_frame, 3, 3), stride=(block_frame, 1, 1), padding=(0, 1, 1))
        self.embed_dim = embed_dim

        # self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=[out_indices])
        self.multiscale_feature = MultiScaleFeature(out_indices=[0, 1, 2, 3])

        # self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, features_only=True,
        #                                   out_indices=[out_indices])

        # self.conv_tmp = nn.Sequential(
        #     nn.Conv2d(160, 1024, 3, 1, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )
        # self.act = self.backbone.act1 if 'act1' in self.backbone else nn.ReLU(inplace=True)
        self.act = nn.ReLU(inplace=True)

        # self.pos_econding = pos_econding
        # if self.pos_econding:
        #     self.PosNet = PosNet(in_plane=64 + 256 + 512 + 1024, out_plane=embed_dim, kernel_size=3)

        # TODO: 3D-pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.pool = nn.AdaptiveAvgPool3d(1)

        self.se = seC3(2048, head_dim, n=1, shortcut=True, g=1, e=0.5)

        self.transformer = Transformer(dim=256,
                                       depth=4,
                                       heads=8,
                                       dim_head=128,
                                       mlp_dim=head_dim,
                                       dropout=drop_rate)

        # self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(drop_rate)
        )

        # self.projection_head = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 128)
        # )

        self.pace = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, self.num_classes)
        )
        self.distype = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, self.distype)
        )

        # 新增失真程度
        self.dislevel = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, self.dis_cd)
        )

    def forward(self, rgb_clip, source_clip=None):
        if source_clip is not None:
            if self.con3d:
                rgb_clip = self.conv3d(rgb_clip)
                source_clip = self.conv3d(source_clip)
                # x = self.act(x)
            # print(f"after con3d : {rgb_clip.shape}")
            B, C, D, H, W = rgb_clip.size()
            # B D C H W
            rgb_clip = rgb_clip.permute(0, 2, 1, 3, 4)
            source_clip = source_clip.permute(0, 2, 1, 3, 4)
            # B*D C H W
            rgb_clip = rgb_clip.contiguous().view(B * D, C, H, W)
            source_clip = source_clip.contiguous().view(B * D, C, H, W)

            # TODO : CNN-model
            # rgb_clip = self.backbone(rgb_clip)[0]
            # source_clip = self.backbone(source_clip)[0]

            rgb_clip = self.multiscale_feature(rgb_clip)  # Tensor(128, 2048)
            source_clip = self.multiscale_feature(source_clip)

            # rgb_clip = self.conv_tmp(rgb_clip)

            # TODO: Transformer
            rgb_clip = self.pool(rgb_clip)
            source_clip = self.pool(source_clip)

            rgb_clip = self.se(rgb_clip)
            source_clip = self.se(source_clip)

            _C, _H, _W = rgb_clip.shape[-3], rgb_clip.shape[-2], rgb_clip.shape[-1]
            rgb_clip = rgb_clip.contiguous().view(B, D, _C)
            source_clip = source_clip.contiguous().view(B, D, _C)

            rgb_clip = self.linear(rgb_clip)
            source_clip = self.linear(source_clip)

            # print(f"batch : {B}, depth : {D}, channel : {}")

            # TODO : transformer
            rgb_clip = self.transformer(rgb_clip)  # tensor(5, 16, 256)
            source_clip = self.transformer(source_clip)  # tensor(5, 16, 256)

            rgb_clip = rgb_clip.mean(dim=1)  # tensor(5, 256)
            source_clip = source_clip.mean(dim=1)  # tensor(5, 256)

            # c_rgb_clip = self.projection_head(rgb_clip)
            # TODO: END
            pace = self.pace(source_clip)
            distype = self.distype(rgb_clip - source_clip)
            # 多加入失真程度
            dislevel = self.dislevel(rgb_clip - source_clip)

            # 多加入返回特征rgb_clip，特征用于对比学习中
            return pace, distype, rgb_clip, dislevel
            # return pace, distype, c_rgb_clip, dislevel

        else:
            if self.con3d:
                rgb_clip = self.conv3d(rgb_clip)
                # x = self.act(x)
            # print(f"after con3d : {rgb_clip.shape}")
            B, C, D, H, W = rgb_clip.size()
            # B D C H W
            rgb_clip = rgb_clip.permute(0, 2, 1, 3, 4)
            # B*D C H W
            rgb_clip = rgb_clip.contiguous().view(B * D, C, H, W)

            # TODO : CNN-model
            # rgb_clip = self.backbone(rgb_clip)[0]
            rgb_clip = self.multiscale_feature(rgb_clip)  # Tensor(128, 2048)

            # rgb_clip = self.conv_tmp(rgb_clip)

            # TODO: Transformer
            rgb_clip = self.pool(rgb_clip)  # 去掉？
            rgb_clip = self.se(rgb_clip)

            _C, _H, _W = rgb_clip.shape[-3], rgb_clip.shape[-2], rgb_clip.shape[-1]
            rgb_clip = rgb_clip.contiguous().view(B, D, _C)

            rgb_clip = self.linear(rgb_clip)

            # print(f"batch : {B}, depth : {D}, channel : {}")

            # TODO : transformer
            rgb_clip = self.transformer(rgb_clip)  # tensor(5, 16, 256)
            rgb_clip = rgb_clip.mean(dim=1)  # tensor(5, 256)

            # c_rgb_clip = self.projection_head(rgb_clip)
            # TODO: END
            pace = self.pace(rgb_clip)
            # dislevel = self.dislevel(rgb_clip)

            # 多加入返回特征rgb_clip，特征用于对比学习中
            # return pace, dislevel, rgb_clip
            return pace, rgb_clip
            # return pace, c_rgb_clip




    # def forward(self, rgb_clip, source_clip=None):
    #     if self.con3d:
    #         rgb_clip = self.conv3d(rgb_clip)
    #         # rgb_org_clip = self.conv3d(source_clip)
    #         # x = self.act(x)
    #     # print(f"after con3d : {rgb_clip.shape}")
    #     B, C, D, H, W = rgb_clip.size()
    #     # B D C H W
    #     rgb_clip = rgb_clip.permute(0, 2, 1, 3, 4)
    #     # rgb_org_clip = rgb_org_clip.permute(0, 2, 1, 3, 4)
    #     # B*D C H W
    #     rgb_clip = rgb_clip.contiguous().view(B * D, C, H, W)
    #     # rgb_org_clip = rgb_org_clip.contiguous().view(B * D, C, H, W)
    #
    #     # TODO : CNN-model
    #     rgb_clip = self.backbone(rgb_clip)[0]
    #     # rgb_clip = self.conv_tmp(rgb_clip)
    #
    #     # TODO: Transformer
    #     rgb_clip = self.pool(rgb_clip)
    #
    #     _C, _H, _W = rgb_clip.shape[-3], rgb_clip.shape[-2], rgb_clip.shape[-1]
    #     rgb_clip = rgb_clip.contiguous().view(B, D, _C)
    #     rgb_clip = self.linear(rgb_clip)
    #     # print(f"batch : {B}, depth : {D}, channel : {}")
    #
    #     # TODO : transformer
    #     rgb_clip = self.transformer(rgb_clip)  # tensor(5, 16, 256)
    #     rgb_clip = rgb_clip.mean(dim=1)  # tensor(5, 256)
    #
    #     # c_rgb_clip = self.projection_head(rgb_clip)
    #     # TODO: END
    #     pace = self.pace(rgb_clip)
    #     distype = self.distype(rgb_clip)
    #     # 多加入失真程度
    #     dislevel = self.dislevel(rgb_clip)
    #
    #     # 多加入返回特征rgb_clip，特征用于对比学习中
    #     return pace, distype, rgb_clip, dislevel
    #
    #     # 多加入返回特征rgb_clip，特征用于对比学习中
    #     # return pace, distype, rgb_clip


