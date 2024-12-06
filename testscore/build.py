# --------------------------------------------------------
# Swin Transformer
# Code Link: https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'


from vb_cnn_transformer import VB_CNN_Transformer
from vb_cnn_transformerv2 import VB_CNN_Transformer2

def build_model(model_type):
    # model_type = config.MODEL.TYPE
    if model_type == "vb_cnn_transformer":
        model = VB_CNN_Transformer(img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=1,
                                    embed_dim=256,  # 96-->256
                                    depths=[2, 2, 6],
                                    num_heads=[3, 6, 12],
                                    window_size=7,
                                    mlp_ratio=4.0,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    ape=False,
                                    drop_rate=0.2,
                                    drop_path_rate=0.2,
                                    attn_drop_rate=0.2,
                                    patch_norm=True,
                                    use_checkpoint=False,
                                    pos_econding=False,
                                    con3d=True,
                                    block_frame=8,
                                    out_indices=3,  # ResNet50 - model
                                    # out_indices=4,  # vgg16 - model
                                    # out_indices=2,  # densetn161 - model
                                    # head_dim=384      # Swin - Transformer
                                    head_dim=1024,  # ResNet50 - model
                                    # head_dim=512,   # vgg16- model
                                    # head_dim=768,     # densetn161- model
                                    imgSize=224,
                                    patchSize=8
                                    )

    elif model_type == "vb_cnn_transformerv2":
        model = VB_CNN_Transformer2(img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=1,
                        embed_dim=256,  # 96-->256
                        depths=[2, 2, 6],
                        num_heads=[3, 6, 12],
                        window_size=7,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        qk_scale=None,
                        ape=False,
                        drop_rate=0.2,
                        drop_path_rate=0.2,
                        attn_drop_rate=0.2,
                        patch_norm=True,
                        use_checkpoint=False,
                        pos_econding=False,
                        con3d=True,
                        block_frame=8,
                        out_indices=3,  # ResNet50 - model
                        # out_indices=4,  # vgg16 - model
                        # out_indices=2,  # densetn161 - model
                        # head_dim=384      # Swin - Transformer
                        head_dim=1024,   # ResNet50 - model
                        # head_dim=512,   # vgg16- model
                        # head_dim=768,     # densetn161- model
                        imgSize=224,
                        patchSize=8
                        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
