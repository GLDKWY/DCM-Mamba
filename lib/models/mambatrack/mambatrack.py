"""
Basic OSTrack + DFM + OFM model.DEBUG
"""
import math
import os
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.mambatrack.vim import vim_tiny_patch16_224, vim_small_patch16_224, vim_middle_patch16_224, vim_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mambatrack.videomamba import videomamba_base, videomamba_middle, videomamba_small, videomamba_tiny
from lib.models.mambatrack.vit import vit_base_patch16_224
from lib.models.fusion.dfm import DFM
from lib.models.fusion.ofm import OFM


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [(input_dim//2**(i+1)) for i in range(num_layers - 1)]
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MambaTrack(nn.Module):
    """ MambaTrack with multi-layer CMIM + SMFM """

    def __init__(self, backbone, box_head, concat_mode="tsts", scan_mode='spatial_first',
                 add_cls_token=False, aux_loss=False, head_type="CORNER",
                 patch8=False, fuse=False):
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head

        # 配置
        self.concat_mode = concat_mode
        self.scan_mode = scan_mode
        self.add_cls_token = add_cls_token
        self.fuse = fuse
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.feat_sz_s = int(box_head.feat_sz) if not patch8 else int(box_head.feat_sz*2)
        self.feat_len_s = self.feat_sz_s**2
        self.mamba_kwargs = {"d_state": 8, "expand": 1, "d_conv": 2}
        embed_dim = backbone.embed_dim
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        if add_cls_token:
            self.iou_head = MLP(backbone.embed_dim, 1, 3)

        # ===== CMIM + SMFM (selected encoder layers 2, 5, 8, 11) =====
        # Lite: single shared CMIM (was 4x per-layer) + lightweight Mamba
        self.use_dfm = True
        self.use_ofm = True
        self.fusion_layers = set([2, 5, 8, 11])
        embed_dim = backbone.embed_dim

        if self.use_dfm:
            self.cmim_shared = DFM(
                dim=self.backbone.embed_dim,
                hidden_mult=2,
                drop=0.1,
                mamba_kwargs=self.mamba_kwargs
            )
            # ModuleDict wrapper: all layers share the same underlying module.
            # NOTE: do NOT assign the same nn.Module to multiple ModuleDict keys
            # (PyTorch would re-parent it and keep only the last key).
            self.dfms = nn.ModuleDict({"shared": self.cmim_shared})
        else:
            self.cmim_shared = None
            self.dfms = None

        if self.use_ofm:
            num_layers = len(self.fusion_layers)
            self.ofm = OFM(embed_dim, num_layers=num_layers, drop=0.0)
        else:
            self.ofm = None
        # ======================

    def forward(self, template: torch.Tensor, search: torch.Tensor, z_frame_ids=None, return_x=False):
        assert isinstance(search, list), "search must be list"
        out_dict = []

        for i in range(len(search)):
            x = self.backbone(
                z=template.copy(),
                x=search[i],
                z_frame_ids=z_frame_ids,
                concat_mode=self.concat_mode,
                scan_mode=self.scan_mode,
                inference_params=None,
                cmims=self.dfms if self.use_dfm else None,
                fusion_layers=self.fusion_layers if self.use_dfm else None,
            )
            if not isinstance(x, list):
                x = [x]

            if self.use_ofm and self.use_dfm and len(x) > 0:
                fused_tokens = self.ofm(x)
                out = self.forward_head_fused(fused_tokens)
            else:
                layers = x
                feat_last = layers[-1]
                tokens = feat_last[:, 1:] if self.add_cls_token else feat_last
                if self.concat_mode == "tsts":
                    all_len = tokens.shape[1]
                    s_rgb = tokens[:, all_len // 2 - self.feat_len_s: all_len // 2]
                    s_X = tokens[:, -self.feat_len_s:]
                elif self.concat_mode in ["ttss", "crosst"]:
                    s_rgb = tokens[:, -2 * self.feat_len_s:-self.feat_len_s]
                    s_X = tokens[:, -self.feat_len_s:]
                else:
                    raise NotImplementedError
                fused_tokens = s_rgb + s_X
                out = self.forward_head(s_rgb, s_X, None)

            if self.add_cls_token and 'pred_iou' not in out:
                pred_iou = self.iou_head(x[-1][:, :1].squeeze()).squeeze()
                out["pred_iou"] = pred_iou

            if return_x:
                out["feat_last"] = x[-1]

            out_dict.append(out)
        return out_dict

    def forward_head_fused(self, fused_tokens):
        s = fused_tokens.unsqueeze(-1).permute(0,3,2,1).contiguous()  # [B,C,1,L]
        bs, Nq, C, HW = s.size()
        s = s.view(-1, C, self.feat_sz_s, self.feat_sz_s)              # [B',C,H,W]
        opt_feat = s

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box).view(bs, Nq, 4)
            return {'pred_boxes': outputs_coord, 'score_map': score_map}
        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, None)
            outputs_coord = bbox.view(bs, Nq, 4)
            return {'pred_boxes': outputs_coord,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
        else:
            raise NotImplementedError
        
    def forward_head(self, s_rgb, s_X, gt_score_map=None):
        """
        s_rgb, s_X: output embeddings of the backbone, it can be (B, Hx*Wx, C)
        """
        srgb = (s_rgb.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = srgb.size()
        s_rgb = srgb.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        sX = (s_X.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = sX.size()
        s_X = sX.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.fuse:
            opt_feat = self.fuse(torch.cat([s_rgb, s_X], dim=1))
        else:
            opt_feat = s_rgb + s_X

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_mambatrack(cfg, training=True):
    add_z_seg = cfg.MODEL.BACKBONE.Z_SEG
    z_vocab_size = cfg.MODEL.BACKBONE.Z_VOCAB_SIZE
    add_cls_token = cfg.MODEL.BACKBONE.ADD_CLS_TOKEN
    # 预训练模型处理
    if 'in1k' in cfg.MODEL.PRETRAIN_FILE:
        pretrained = cfg.MODEL.PRETRAIN_FILE
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vim_tiny_patch16_224':
        backbone = vim_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vim_small_patch16_224':
        backbone = vim_small_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vim_small_patch16_432':
        backbone = vim_small_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=432)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vim_middle_patch16_224':
        backbone = vim_middle_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vim_base_patch16_224':
        backbone = vim_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim

    # ---------------------------------------
    # default resolution is 224      add_z_seg z_vocab_size
    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_tiny_576':
        backbone = videomamba_tiny(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=576,
        add_z_seg=add_z_seg, z_vocab_size=z_vocab_size, add_cls_token=add_cls_token)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_small_576':
        backbone = videomamba_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=576,
        add_z_seg=add_z_seg, z_vocab_size=z_vocab_size, add_cls_token=add_cls_token)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_middle':
        backbone = videomamba_middle(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        add_z_seg=add_z_seg, z_vocab_size=z_vocab_size, add_cls_token=add_cls_token)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_middle_576':
        backbone = videomamba_middle(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=576,
        add_z_seg=add_z_seg, z_vocab_size=z_vocab_size, add_cls_token=add_cls_token)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_base':
        backbone = videomamba_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        add_z_seg=add_z_seg, z_vocab_size=z_vocab_size, add_cls_token=add_cls_token)
        hidden_dim = backbone.embed_dim
    # ---------------------------------------
    elif cfg.MODEL.BACKBONE.TYPE == 'ostrack_base':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim



    else:
        raise NotImplementedError

    if 'videomamba' in cfg.MODEL.BACKBONE.TYPE or 'vim' in cfg.MODEL.BACKBONE.TYPE:
        backbone.interpolate_pos_embed(cfg=cfg)
    elif 'ostrack' in cfg.MODEL.BACKBONE.TYPE:
        backbone.finetune_track(cfg, 1)

    box_head = build_box_head(cfg, hidden_dim, patch8="patch8" in cfg.MODEL.BACKBONE.TYPE)

    model = MambaTrack(
        backbone,
        box_head,
        concat_mode=cfg.MODEL.BACKBONE.CONCAT_MODE,
        scan_mode=cfg.MODEL.BACKBONE.SCAN_MODE,
        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        patch8="patch8" in cfg.MODEL.BACKBONE.TYPE,
        fuse= cfg.MODEL.HEAD.FUSE
    )

    if ('OSTrack_videomamba' in cfg.MODEL.PRETRAIN_FILE or 'OSTrack_vim' in cfg.MODEL.PRETRAIN_FILE) and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model missing_keys: ', missing_keys)
        print('Load pretrained model unexpected_keys: ', unexpected_keys)

    return model

if __name__ == '__main__':
    import importlib

    config_module = importlib.import_module("lib.config.%s.config" % 'mambatrack')
    cfg = config_module.cfg
    config_module.update_config_from_file('')
    ostrack = build_mambatrack(cfg, training=True)
    # print(mambatrack.backbone)