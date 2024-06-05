"""
2024.01.30
执行定量分析，语义分割，得到我们的点云
"""
import sys
sys.path.append("/code1/dyn/github_repos/OpenGraph")
import copy
import json
import os
import pickle
import gzip
import argparse
import random
import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import hydra
import cv2
from omegaconf import DictConfig
import distinctipy
from some_class.map_calss import MapObjectList
from utils.utils import  load_models
import json
from PIL import Image
from pathlib import Path
from tqdm import trange


def merge_point_clouds_with_labels(pcds, labels, class_colors_sk_disk):
    """
    Merge a list of point clouds with corresponding labels into a single point cloud.
    Args:
        pcds (List[o3d.geometry.PointCloud]): List of point clouds.
        labels (List[np.ndarray]): List of labels corresponding to each point cloud.
    Returns:
        o3d.geometry.PointCloud: Merged point cloud with labels.
    """
    merged_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        class_label = labels[i]
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        color = np.array(class_colors_sk_disk[class_label])/255.0
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
        merged_pcd += pcd
    return merged_pcd


def load_result(result_path):
    '''
    加载各个对象
    '''
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])
        instance_colors = distinctipy.get_colors(len(objects)+len(bg_objects), pastel_factor=0.5)  
        instance_colors = {str(i): c for i, c in enumerate(instance_colors)}
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
        instance_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)  # 生成一组视觉上相异的颜色
        instance_colors = {str(i): c for i, c in enumerate(instance_colors)}
    else:
        raise ValueError("Unknown results type: ", type(results))
    return objects, bg_objects, instance_colors



def load_colors(cfg):
    '''
    类别语义颜色文件
    '''
    file_path = cfg.class_colors_json
    with open(file_path, 'r') as json_file:
        class_colors_sk_disk = json.load(json_file)
    return class_colors_sk_disk
    
@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    assert not (cfg.result_path is None), \
        "Either result_path must be provided."
    # 加载pcd结果
    objects, bg_objects, instance_colors = load_result(cfg.result_path)
    class_colors_sk_disk = load_colors(cfg)
    # 如果存在背景物体，放在后面一部分
    if bg_objects is not None:
        indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        objects.extend(bg_objects)
    # 下采样点云
    for i in trange(len(objects)):
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(0.1)
        objects[i]['pcd'] = pcd
    pcds = copy.deepcopy(objects.get_values("pcd"))
    class_labels = copy.deepcopy(objects.get_values("class_sk"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # 得到全局带标签的点云
    full_point_cloud_with_labels = merge_point_clouds_with_labels(pcds, class_labels, class_colors_sk_disk) 
    full_point_cloud_with_labels = full_point_cloud_with_labels.voxel_down_sample(0.1)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(full_point_cloud_with_labels)
    # for bbox in bboxes:
    #     vis.add_geometry(bbox) 
    # vis.run()
    # o3d.visualization.draw_geometries([full_point_cloud_with_labels])
    # 保存点云
    print("Save our pcd to", cfg.our_pcd)
    o3d.io.write_point_cloud(cfg.our_pcd, full_point_cloud_with_labels)
    
if __name__ == "__main__":
    main()