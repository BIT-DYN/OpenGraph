"""
2024.01.18 
加载semantic数据格式的一个类
"""

import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted


class SemanticKittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        **kwargs,
    ):
        # 序列的基本文件
        self.input_folder = os.path.join(basedir, sequence)
        # 标定文件，直接加载
        self.calib_path = os.path.join(self.input_folder, "calib.txt")
        self.calib = self.load_calib()
        # pose文件，直接把pose全读出来
        self.pose_path = os.path.join(self.input_folder, "poses.txt")
        self.poses = self.load_poses()
        # 图像文件，读取image_2
        self.color_paths = natsorted(glob.glob(f"{self.input_folder}/image_2/*.png"))
        # 点云文件，读取velodyne
        self.pc_paths = natsorted(glob.glob(f"{self.input_folder}/velodyne/*.bin"))
        # 开始id和结束id，没有则默认全部
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )
        # 看看是不是读对了
        if len(self.color_paths) != len(self.pc_paths):
            raise ValueError("Number of color and depth images must be the same.")
        self.num_imgs = len(self.color_paths)
        if self.end == -1:
            self.end = self.num_imgs
        # 保持所有的poses和pc_paths以多帧重叠投影
        self.all_pc_paths = self.pc_paths
        self.all_poses = self.poses
        # 按照stride的间隔读取
        self.stride = stride
        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.pc_paths = self.pc_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # 此时的文件长度
        self.num_imgs = len(self.color_paths)
        print("\n Congratulations! The SemanticKITTI dataset is loaded and ready for any use! \n")
        super().__init__()

    def load_poses(self):
        '''
        加载位姿
        '''
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
            poses = np.array([list(map(float, line.strip().split())) for line in lines])
            poses = poses.reshape(-1,3,4)
            ones_column = np.zeros((poses.shape[0], 1, 4))
            ones_column[:, :, -1] = 1.0
            poses = np.append(poses, ones_column, axis=1)
            # 变换到雷达坐标系
            poses = poses @ self.calib['T_cam2_velo']
        return poses
    
    def load_calib(self):
        '''
        加载标定文件
        '''
        calib = {}
        with open(self.calib_path, "r") as calib_file:
            calib_lines = calib_file.readlines()
            # 加载相机内参
            P_rect_line = calib_lines[2]
            P_rect_02 = np.array(list(map(float, P_rect_line.strip().split()[1:]))).reshape(3, 4)
            calib["P_rect_20"] = P_rect_02
            # 加载相机外参
            Tr_line = calib_lines[4]
            Tr = np.array(list(map(float, Tr_line.strip().split()[1:]))).reshape(3, 4)
            Tr = np.vstack([Tr, [0, 0, 0, 1]])
            calib['T_cam2_velo'] = Tr
        return calib
    
    def load_velo_scan(self, velo_filename):
        '''
        加载雷达数据
        '''
        scan = np.fromfile(velo_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan
    
    def __len__(self):
        '''
        得到数据集中数据的数量
        '''
        return self.num_imgs
    
    def __getitem__(self, index):
        '''
        索引，按照索引值数据类的一个数据，包括图像、点云、位姿
        '''
        color_path = self.color_paths[index]
        pc_path = self.pc_paths[index]
        # cv图像格式
        color = cv2.imread(color_path)
        # np格式
        pointCloud = self.load_velo_scan(pc_path)  # 读取lidar原始数据
        pose = self.poses[index]
        # 历史多帧的重叠投影
        his_pointCloud = []
        his_pose = []
        if index > 0:
            for i in range(self.stride-1):
                his_index = self.start+index*self.stride-i-1
                his_pointCloud.append(self.load_velo_scan(self.all_pc_paths[his_index]))
                his_pose.append(self.all_poses[his_index])
        return (
            color,
            pointCloud,
            pose,
            his_pointCloud,
            his_pose
        )
