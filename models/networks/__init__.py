from models.networks.nestedT import Nest
from models.networks.swin_unet import SwinTransformerSys
from models.networks.swin_transformer import SwinTransformer
from models.networks.axialnet import axialunet, logo,MedT,gated
from models.networks.vit_seg_modeling_gate import VisionTransformer_AG
from .unet_2D import *
from .unet_3D import *
from .unet_nonlocal_2D import *
from .unet_nonlocal_3D import *
from .unet_grid_attention_3D import *
from .unet_CT_dsv_3D import *
from .unet_CT_single_att_dsv_3D import *
from .unet_CT_multi_att_dsv_3D import *
from .unet_CT_multi_att_dsv_2D import *
from .sononet import *
from .sononet_grid_attention import *
from .vit_seg_modeling import *
import pywick.models.segmentation as pws
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from timm.models.swin_transformer import _create_swin_transformer


def get_network(name,config, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2),
                aggregation_mode='concat',img_size=256):
    model = _get_model_instance(name, tensor_dim)

    if name in ['unet', 'unet_ct_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False)
    elif name in ['unet_nonlocal']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      is_deconv=False,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale)
    elif name in ['unet_grid_gating',
                  'unet_ct_single_att_dsv',
                  'unet_ct_multi_att_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False)
    elif name in ['sononet','sononet2']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale)
    elif name in ['sononet_grid_attention']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      nonlocal_mode=nonlocal_mode,
                      aggregation_mode=aggregation_mode)
    elif name in ['DeepLab','Deeplab']:
        model = model(num_classes=n_classes,pretrained=True,in_channels = in_channels)
    elif name == "axialunet":
        model = axialunet(img_size= img_size, imgchan = n_classes)
    elif name == "nest":
        model = model(img_size= img_size, patch_size=config.patch_size,num_classes=n_classes,embed_dims=config.embed_dims,num_heads=config.num_heads,depths=config.depths,num_levels=config.num_levels)
    elif name == "medt":
        model = MedT(img_size= img_size, num_classes = n_classes)
    elif name== "gatedaxialunet":
        model = gated(img_size= img_size, num_classes = n_classes)
    elif name == "swin":
        model = model(img_size= img_size,patch_size=config.patch_size,in_chans=in_channels,num_classes=n_classes,embed_dim=config.embed_dim,depths=config.depth,num_heads=config.num_heads,window_size=config.window_size)
        # model = model(img_size= img_size,num_classes=n_classes)

    elif name == "swin_class":
        model_kwargs = dict(
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=n_classes,
    )
        model =  _create_swin_transformer(
        "swin_large_patch4_window7_224", pretrained=False, **model_kwargs
    )
    elif name == "swin_unet":
        model = model(img_size= img_size,patch_size=config.patch_size,in_chans=in_channels,num_classes = n_classes,embed_dim=config.embed_dim,depths=config.depth,num_heads=config.num_heads,window_size=config.window_size)
    elif name== "logo":
        model = logo(img_size= img_size, num_classes = n_classes)
    elif name in CONFIGS_ViT_seg.keys():
        config_vit = CONFIGS_ViT_seg[name]
        config_vit.n_classes = n_classes
        img_size = config_vit.img_size if hasattr(config_vit,"img_size") else img_size
        if 'R50' in name:
            config_vit.patches.grid = (int(img_size / config_vit.patches.grid[0]), int(img_size / config_vit.patches.grid[1]))
    
        model = model(config_vit,img_size = img_size , num_classes=n_classes)
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'unet':{'2D': unet_2D, '3D': unet_3D},
        'unet_nonlocal':{'2D': unet_nonlocal_2D, '3D': unet_nonlocal_3D},
        'unet_grid_gating': {'3D': unet_grid_attention_3D},
        'unet_ct_dsv': {'3D': unet_CT_dsv_3D},
        'unet_ct_single_att_dsv': {'3D': unet_CT_single_att_dsv_3D},
        'unet_ct_multi_att_dsv': {'3D': unet_CT_multi_att_dsv_3D,"2D": unet_CT_multi_att_dsv_2D},
        'sononet': {'2D': sononet},
        'sononet2': {'2D': sononet2},
        'sononet_grid_attention': {'2D': sononet_grid_attention},
        'ViT-B_16': {'2D':VisionTransformer},
        'ViT-B_32': {'2D':VisionTransformer},
        'ViT-L_16': {'2D':VisionTransformer},
        'ViT-H_14': {'2D':VisionTransformer},
        'medt': {'2D':MedT},
        'gatedaxialunet': {'2D':gated},
        'axialunet': {'2D':axialunet},
        'logo': {'2D':logo},
        'R50-ViT-B_32': {'2D':VisionTransformer},
        'R50-ViT-B_16': {'2D':VisionTransformer},
        'R50-ViT-B_16_AG': {'2D':VisionTransformer_AG},
        'swin': {'2D':SwinTransformer,"class":"class"},
        'swin_class': {'2D':SwinTransformer,},
        'swin_unet': {'2D':SwinTransformerSys},
        'R50-ViT-L_16': {'2D':VisionTransformer},
        'nest': {'2D':Nest},
        'DeepLab':{'v3+':pws.deeplab_v3_plus.DeepLabv3_plus,"v2":pws.deeplab_v2_res.DeepLabv2_ASPP,"v3":pws.deeplab_v3.DeepLabv3}

    }[name][tensor_dim]
