import torch
import os
import imgviz
import PIL.Image
import time
import numpy as np
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
from seg_model import SegModel

root = os.getcwd()
config_file = root + f'/model_config/mask_rcnn_r50_fpn_v3_c.py'
checkpoint_file = root + f'/model_config/model/mask_rcnn_r50_fpn_v3_c.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SegModel(config_file, checkpoint_file)

img_big = r'E:\DL task\data_stone\data\2022-10-11\aggregate_2022-10-11-10-32-17_1.jpg'
img_files = [r'E:\DL task\data_stone' + r'\coco\train\JPEGImages\aggregate_2022-10-11-10-32-15_1_c.jpg',
             r'E:\DL task\data_stone' + r'\coco\train\JPEGImages\aggregate_2022-10-11-10-32-17_1_c.jpg',
             r'E:\DL task\data_stone' + r'\coco\train\JPEGImages\aggregate_2022-10-11-10-32-18_1_c.jpg',
             r'E:\DL task\data_stone' + r'\coco\train\JPEGImages\aggregate_2022-10-11-10-32-19_1_c.jpg']

running_time = []
for i in range(10):
    start = time.time()
    model.inference_detector(img_files)
    end = time.time()
    running_time.append(end - start)
avg_running_time = (sum(running_time) - max(running_time) - min(running_time)) / 8
print('time cost : %.3f sec' % avg_running_time)

