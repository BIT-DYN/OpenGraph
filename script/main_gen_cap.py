"""
2024.01.17 
传入三个大模型，得到数据集中的图像的caption和mask
"""
import sys
sys.path.append("/code1/dyn/github_repos/OpenGraph")
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
import glob
import os
import gzip
import pickle
from utils.utils import  load_models, project
from tqdm import trange
from some_class.datasets_class import SemanticKittiDataset
import open3d as o3d
import hydra
from omegaconf import DictConfig


def process_cfg(cfg: DictConfig):
    '''
    配置文件预处理
    '''
    cfg.basedir = Path(cfg.basedir)
    cfg.save_vis_path = Path(cfg.save_vis_path)
    cfg.save_cap_path = Path(cfg.save_cap_path)
    return cfg


@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    # 先处理一下cfg
    cfg = process_cfg(cfg)
    # 加载用到的大模型们，可以实现分割
    mask_generator = load_models(cfg)
    # 加载所使用的数据集
    datasets = SemanticKittiDataset(cfg.basedir, cfg.sequence, stride=cfg.stride, start=cfg.start, end=cfg.end)
    print("Load a dataset with a size of:", len(datasets))
    for idx in trange(len(datasets)):
        image, _, _, _, _ = datasets[idx]
        file_names = os.path.basename(datasets.color_paths[idx])
        save_path_vis = Path(os.path.join(cfg.save_vis_path, f"vis_{file_names}"))
        # 保存为离线，太占用时间啦
        masks_result = mask_generator.generate(image, save_path=save_path_vis, save_vis=cfg.save_vis)
        if cfg.save_cap:
            save_path_cap = Path(os.path.join(cfg.save_cap_path, f"cap_{file_names}")).with_suffix(".pkl.gz")
            # 如果目录不存在则创建
            save_path_cap.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(save_path_cap, "wb") as f:
                pickle.dump(masks_result, f)

if __name__ == "__main__":
    main()           
    

