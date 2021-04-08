import mmcv
import os.path as osp

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
from mmcv.runner import set_random_seed
from mmcv import Config

import pandas as pd
object = pd.read_pickle(r'/home/kiki/Documents/Cercetare/mmaction2/data/ucf101_24/annotations/UCF101v2-GT.pkl')

cfg = Config.fromfile('/home/kiki/Documents/Cercetare/mmaction2/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py')

#print(f'Config:\n{cfg.pretty_text}')
print(cfg.data.videos_per_gpu)

"""

python demo/demo_spatiotemporal_det.py 
--video demo/demo.mp4 --out-filename demo/demo_out_st.mp4 
--config configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py 
--checkpoint checkpoints/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217-ae225e97.pth  
--det-config demo/faster_rcnn_r50_fpn_2x_coco.py  
--det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth 
--det-score-thr 0.9  
--action-score-thr 0.5 
--label-map demo/label_map_ava.txt 
--predict-stepsize 8  
--output-stepsize 4 
--output-fps 6
"""
#cfg.label_file = object['labels']
#cfg.ann_file_train = None
#cfg.ann_file_val = None
cfg.setdefault('omnisource', False)
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)