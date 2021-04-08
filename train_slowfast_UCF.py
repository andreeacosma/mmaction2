import mmcv
import os.path as osp

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
from mmcv.runner import set_random_seed
from mmcv import Config

import pandas as pd
object = pd.read_pickle(r'/home/kiki/Documents/Cercetare/mmaction2/data/ucf101_24/annotations/UCF101v2-GT.pkl')

cfg = Config.fromfile('/home/kiki/Documents/Cercetare/mmaction2/configs/detection/ucf101-24/slowfast_kinetics_pretrained_r50_4x16x1_20e_ucf_rgb.py')

#print(f'Config:\n{cfg.pretty_text}')
print(cfg.data.videos_per_gpu)

cfg.proposal_file_train=object['gttubes']
cfg.proposal_file_val=object['gttubes']


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