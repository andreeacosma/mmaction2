import mmcv
import os.path as osp
import argparse

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
from mmcv.runner import set_random_seed
from mmcv import Config

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--dataset', help='type of dataset eg: AVA, UCF')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Reading config")
    if args.config is not None:
        try:
            cfg = Config.fromfile(args.config)
            # cfg = Config.fromfile('/home/kiki/Documents/Cercetare/mmaction2/configs/detection/ucf101-24/slowfast_kinetics_pretrained_r50_4x16x1_20e_ucf_rgb.py')
        except Exception as e:
            print("Config file not found")
            exit()
    print("Videos per gpu: {}". format(cfg.data.videos_per_gpu))
    #print(f'Config:\n{cfg.pretty_text}')
    if args.dataset=="UCF":
        object = pd.read_pickle(r'./data/ucf101_24/annotations/UCF101v2-GT.pkl')
        cfg.proposal_file_train=object['gttubes']
        cfg.proposal_file_val=object['gttubes']


    cfg.setdefault('omnisource', False)
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Build the dataset
    print("Building dataset")
    datasets = [build_dataset(cfg.data.train)]

    # Build the recognizer
    print("Building model")
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    print("Start training\n")
    train_model(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()