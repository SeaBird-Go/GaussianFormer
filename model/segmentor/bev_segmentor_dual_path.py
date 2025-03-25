'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-02-19 17:09:45
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the dual path for GuassianFormer that includes the pretrained 
query prediction head and the GaussianFormer itself.
'''
import torch, time
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone

from .base_segmentor import CustomBaseSegmentor


@SEGMENTORS.register_module()
class BEVSegmentorDualPath(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        # use_post_fusion=False,
        ssp_model=None,
        freeze_ssp=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        # self.use_post_fusion = use_post_fusion
        if ssp_model is not None:
            self.ssp_model = SEGMENTORS.build(ssp_model)

        if freeze_ssp and hasattr(self, "ssp_model"):
            self.ssp_model.requires_grad_(False)
        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                out_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        ## 1) Forward the sparse splating pre-training (SSP) model
        self.ssp_model.eval()
        with torch.no_grad():
            ssp_results = self.ssp_model(imgs=imgs, metas=metas, 
                                         out_only=True, points=points, **kwargs)
            
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)

        # Update the results with the SSP results
        results.update({f"ssp_{key}": value for key, value in ssp_results.items()})

        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        outs = self.lifter(**results)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # results.update({"lifter_time": elapsed})

        results.update(outs)

        outs = self.encoder(**results)
        if out_only:
            return outs
        if rep_only:
            return outs['representation']
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results