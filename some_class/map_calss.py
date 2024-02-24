
'''
2024.01.18 
点云地图的类，支持各种操作
'''
from collections.abc import Iterable
import copy
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d


class DetectionList(list):
    '''
    用于暂时存储某个相机帧内的物体类别
    '''
    def get_values(self, key, idx:int=None):
        '''
        得到某个物体的某个类别
        '''
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        '''
        得到堆叠起来的值torch类型
        '''
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            # 如果是box类型，要先变换为box上的点
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            # 所有np类型转为torch
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        '''
        得到堆叠起来的值numpy类型
        '''
        values = self.get_stacked_values_torch(key, idx)
        from utils.utils import to_numpy
        return to_numpy(values)
    
    def get_stacked_str_torch(self, key, idx:int=None):
        '''
        得到堆叠起来的caption类型
        '''
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            values.append(v)
        return values
    
    def __add__(self, other):
        '''
        复制一个新的列表，再加入其他值
        '''
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        '''
        不复制，只增加
        '''
        self.extend(other)
        return self
    
    def slice_by_indices(self, index: Iterable[int]):
        '''
        通过索引返回当前列表的子列表
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self
    
    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        通过mask返回当前列表的子列表
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self
    
                
    def color_by_instance(self):
        '''
        按照instance确定颜色
        '''
        if len(self) == 0:
            return
        # 如果有内设的颜色，直接将点云设为该颜色
        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        # 否则每个就随机设置，我们按照这种即可
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]
            
    
class MapObjectList(DetectionList):
    '''
    用于存储整个点云列表
    '''
    def compute_similarities(self, new_ft):
        '''
        输入新的点云的特征，计算相似性
        '''
        # 如果是一个 numpy 数组，则使其成为一个张量
        from utils.utils import to_tensor
        new_ft = to_tensor(new_ft)
        # 假设计算特征的余弦相似性，则需要得到所有实例的
        clip_fts = self.get_stacked_values_torch('ft')
        # 计算相似性
        similarities = F.cosine_similarity(new_ft.unsqueeze(0), clip_fts)
        # 返回一个相似性数值
        return similarities
    
    def to_serializable(self):
        '''
        序列，得到最简单形式的numpy，方便存储
        '''
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            from utils.utils import to_numpy
            s_obj_dict['ft'] = to_numpy(s_obj_dict['ft'])
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            # 把pcd和bbox直接删掉，只保留其中的点和bbox的点
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            s_obj_list.append(s_obj_dict)
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        '''
        加载序列
        '''
        # 对于被加载的序列需要是空的
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)
            # 把键值复制回来
            from utils.utils import to_tensor
            new_obj['ft'] = to_tensor(new_obj['ft'])
            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
            # 删掉用不到的键值
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']
            self.append(new_obj)