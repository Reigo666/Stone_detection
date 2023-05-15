import os
import time
import math
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from mmdet.apis import init_detector
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from config import my_config
import mmcv

cfg_img = [
    dict(type='LoadImageFromWebcam'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

cfg_img_path = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]


class SegModel:
    def __init__(self, model_config, model_checkpoint):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = init_detector(model_config, model_checkpoint, device=device)
        self.test_pipeline_img = Compose(cfg_img)
        self.test_pipeline_img_path = Compose(cfg_img_path)
        self.scale = my_config.config['camera']['scale']
        self.roundness_thr = my_config.config['camera']['roundness_thr']

    def inference_detector(self, imgs):
        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False
        #
        device = next(self.model.parameters()).device  # model device
        test_pipeline = self.test_pipeline_img
        if not isinstance(imgs[0], np.ndarray):
            test_pipeline = self.test_pipeline_img_path

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]

        # forward the model
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)

        if not is_batch:
            return results[0]
        else:
            return results

    def deal_result(self, img_file, result, score_thr=0.5, save_path='runs/2.1_rrat/add/'):
        bbox_result, segm_result = result
        bboxes = np.vstack(bbox_result)
        # labels = [
        #     np.full(bbox.shape[0], i, dtype=np.int32)
        #     for i, bbox in enumerate(bbox_result)
        # ]
        # labels = np.concatenate(labels)
        segms = mmcv.concat_list(segm_result)
        segms = np.stack(segms, axis=0)

        img = mmcv.imread(img_file, channel_order='rgb').astype(np.uint8)
        img = np.ascontiguousarray(img)
        width, height = img.shape[1], img.shape[0]

        fig = plt.figure('', frameon=False)
        plt.title('')
        canvas = fig.canvas
        dpi = fig.get_dpi()
        fig.set_size_inches((width + 1e-2) / dpi, (height + 1e-2) / dpi)

        # remove white edges by set subplot margin
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        if score_thr > 0:
            scores = bboxes[:, -1]
            scores = np.around(scores, 2)
            # 按得分降序排列
            inds = np.argwhere(scores >= score_thr)[:, 0]
            scores = scores[inds]
            scores_inds = list(zip(inds, scores))
            scores_inds.sort(key=lambda x: x[1], reverse=True)
            inds_sorted = [x[0] for x in scores_inds]
            # bboxes = bboxes[inds_sorted, :]
            # labels = labels[inds_sorted]
            segms = segms[inds_sorted, ...]

        masks = segms
        # 为每个掩码生成颜色
        taken_colors = {0, 0, 0}
        random_colors = np.random.randint(0, 255, (masks.shape[0], 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)

        predict_nums = np.zeros(4)
        predict_area = np.zeros(4)
        total_length = 0
        max_length = 0
        positions = []
        # 圆度低于阈值计数
        roundness_sum = 0
        roundness_area = 0
        repeat = 0
        mask_bg = np.full((height, width), 2)
        for i, mask in enumerate(masks):
            # 计算覆盖率，重复像素数
            mask_uint = mask.astype(np.uint8)
            this_repeat = np.sum((mask_bg == mask_uint))
            # 与已测出区域重复超过50%则忽略
            if mask_uint.sum() != 0 and this_repeat / mask_uint.sum() >= 0.5:
                continue
            mask_bg[mask_uint.astype(bool)] = 1
            # 计算最大连通区域
            _, labels_c, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=4)
            if len(centroids) == 1:
                continue
            largest_id = np.argmax(stats[1:, -1]) + 1
            positions.append(centroids[largest_id])
            labels_c[labels_c != largest_id] = 0
            labels_c[labels_c == largest_id] = 1
            # 计算面积
            area = stats[largest_id][-1]
            # 计算圆度和粒径
            contours, _ = cv2.findContours(labels_c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 1:
                p = cv2.arcLength(contours[0], True)
                # 圆度按以下方式计算 4 * pi * 面积 / 周长^2
                roundness = 4 * math.pi * area / p ** 2
                if roundness < self.roundness_thr:
                    roundness_sum += 1
                    roundness_area += area
            # 计算等效粒径
            length1 = round(math.sqrt(area / math.pi), 1) * 2
            # 计算外接圆直径
            (x, y), radius = cv2.minEnclosingCircle(contours[0])
            length2 = radius * 2
            if length2 > 100:
                length = 0.85 * length1 + 0.15 * length2
            else:
                length = 0.15 * length1 + 0.85 * length2
            length = length / self.scale
            predict_nums[_get_rank(length)] += 1
            predict_area[_get_rank(length)] += area
            # 用于计算平均粒径
            total_length += length
            # 计算最大粒径
            max_length = max(max_length, length)
            color_mask = color[i]
            while tuple(color_mask) in taken_colors:
                color_mask = _get_bias_color(color_mask)
            taken_colors.add(tuple(color_mask))

            mask = labels_c.astype(bool)
            img[mask] = img[mask] * (1 - 0.8) + color_mask * 0.8
        # 标出目标得分
        # img = Image.fromarray(img)
        # draw = ImageDraw.Draw(img)
        # for idx, pos in enumerate(positions):
        #     draw.text((pos[0], pos[1]), f'|{bboxes[idx - 1][4]:.02f}')
        # img = np.array(img)
        # 保存图像
        # name = os.path.basename(img_file)
        # mmcv.imwrite(img, save_path + name)
        return dict(
            img=img,
            predict_nums=predict_nums,
            predict_area=predict_area,
            roundness_sum=roundness_sum,
            roundness_area=roundness_area,
            total_length=total_length,
            max_length=max_length,
        )


def _get_bias_color(base, max_dist=30):
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def _get_rank(length):
    if length <= my_config.config['rank']['rank1']:
        return 0
    elif length <= my_config.config['rank']['rank2']:
        return 1
    elif length <= my_config.config['rank']['rank3']:
        return 2
    else:
        return 3
