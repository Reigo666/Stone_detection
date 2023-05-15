# Copyright (c) OpenMMLab. All rights reserved.
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class FCNMaskScoreHead(BaseModule):

    def __init__(self,
                 loss_iou=dict(
                     type='MSELoss',
                     loss_weight=0.5),
                 num_classes=1,
                 init_cfg=None):
        super(FCNMaskScoreHead, self).__init__(init_cfg)
        self.loss_iou = build_loss(loss_iou)
        self.num_classes = num_classes
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=257, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=7 * 7 * 32, out_features=196),
        #     nn.Linear(in_features=196, out_features=2),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=14 * 14 * 2, out_features=196),
            nn.Linear(in_features=196, out_features=2),
        )
        self.maxPool = nn.Sequential(nn.MaxPool2d(kernel_size=2))

    @auto_fp16()
    def forward(self, mask_pred, mask_feats):
        # mask_pred = mask_pred.unsqueeze(1)
        # mask_pred = self.maxPool(mask_pred)
        # # print(mask_pred.shape)
        # # print(mask_feats.shape)
        # mask_iou_feat = torch.cat((mask_pred, mask_feats), dim=1)
        # mask_iou_pred = self.conv(mask_iou_feat)
        # mask_iou_pred = mask_iou_pred.flatten(1)
        # mask_iou_pred = self.fc(mask_iou_pred)
        mask_pred = mask_pred.unsqueeze(1)
        mask_pred = self.maxPool(mask_pred)
        # print(mask_pred.shape)
        # print(mask_feats.shape)
        mask_feats = self.conv1(mask_feats)
        mask_iou_feat = torch.cat((mask_pred, mask_feats), dim=1)
        mask_iou_pred = self.conv2(mask_iou_feat)
        mask_iou_pred = mask_iou_pred.flatten(1)
        mask_iou_pred = self.fc(mask_iou_pred)
        return mask_iou_pred

    @force_fp32(apply_to=('mask_iou_pred',))
    def loss(self, mask_iou_pred, mask_iou_targets):
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(mask_iou_pred[pos_inds],
                                          mask_iou_targets[pos_inds])
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)

    @force_fp32(apply_to=('mask_pred',))
    def get_targets(self, sampling_results, gt_masks, mask_pred, mask_targets,
                    rcnn_train_cfg):
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (BitmapMask | PolygonMask): Gt masks (the whole instance)
                of each image, with the same shape of the input image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        # compute the area ratio of gt areas inside the proposals and
        # the whole instance
        area_ratios = map(self._get_area_ratio, pos_proposals,
                          pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_pred = (mask_pred > 0.5).float()
        mask_pred_areas = mask_pred.sum((-1, -2))

        # mask_pred and mask_targets are binary maps
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))

        # compute the mask area of the whole instance
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)

        mask_iou_targets = overlap_areas / (
                mask_pred_areas + gt_full_areas - overlap_areas)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance."""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # compute mask areas of gt instances (batch processing for speedup)
            gt_instance_mask_area = gt_masks.areas
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                # crop the gt mask inside the proposal
                bbox = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask.crop(bbox)

                ratio = gt_mask_in_proposal.areas[0] / (
                        gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
                pos_proposals.device)
        else:
            area_ratios = pos_proposals.new_zeros((0,))
        return area_ratios

    @force_fp32(apply_to=('mask_iou_pred',))
    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = mask_iou_pred[inds, det_labels]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [mask_scores[det_labels == i] for i in range(self.num_classes)]
