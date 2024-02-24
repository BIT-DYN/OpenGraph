"""
2024.01.18 
读取离线的caption和mask，映射到点云
"""
import sys
sys.path.append("/home/dyn/outdoor/omm")
import numpy as np
import cv2
import torch
from pathlib import Path
import os
import gzip
import pickle
from sentence_transformers import SentenceTransformer
from datetime import datetime
from utils.utils import project, gobs_to_detection_list, denoise_objects, filter_objects, merge_objects, accumulate_pc, distance_filter, show_captions, class_objects
from tqdm import trange
from some_class.datasets_class import SemanticKittiDataset
from some_class.map_calss import MapObjectList
from utils.merge import compute_spatial_similarities, compute_caption_similarities, compute_ft_similarities, aggregate_similarities, merge_detections_to_objects, caption_merge, captions_ft
import open3d as o3d
import hydra
from omegaconf import DictConfig
from utils.merge import merge_obj2_into_obj1
import mos4d.models.models_test as models

# 一些背景常见的caption：道路，人行道
# 背景可能产生的标签
BG_CAPTIONS = ["paved city road", "a long narrow street", "paved city street", \
    "white lines on the road", "shadows on the street", "a concerte sidewalk", "a shadow on the ground", \
    "white lines on the street", "brick sidewalk", "a sidewalk next to the street",\
    "a train boarding platform", "the train tracks", "shadow of fence",\
    "a sidewalk next to the train tracks", "shadow of bench", "a paved city sidewalk"]
# 背景映射出来的标签
BG_CAPTIONS_Pro = ["paved road", "paved road", "paved road",\
    "paved road", "paved road", "paved road", "paved road", \
    "paved road", "sidewalk", "sidewalk",\
    "sidewalk", "sidewalk", "sidewalk", \
    "sidewalk", "sidewalk", "sidewalk"]
# 背景映射出来的标签有哪些
BG_CAPTIONS_Pro_Sim = ["paved road", "sidewalk"]


def process_cfg(cfg: DictConfig):
    '''
    配置文件预处理
    '''
    cfg.basedir = Path(cfg.basedir)
    cfg.save_vis_path = Path(cfg.save_vis_path)
    cfg.save_cap_path = Path(cfg.save_cap_path)
    cfg.save_pcd_path = Path(cfg.save_pcd_path)
    return cfg

@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    # 先处理一下cfg
    cfg = process_cfg(cfg)
    # 加载所使用的数据集
    datasets = SemanticKittiDataset(cfg.basedir, cfg.sequence, stride=cfg.stride, start=cfg.start, end=cfg.end)
    print("Load a dataset with a size of:", len(datasets))
    # 初始化地图
    objects = MapObjectList(device="cuda")
    # 是否过滤动态目标，如果需要就加载模型
    mos_model = None
    if cfg.filter_dynamic:
        weights = cfg.mos_path
        mos_cfg = torch.load(weights)["hyper_parameters"]
        ckpt = torch.load(weights)
        mos_model = models.MOSNet(mos_cfg)
        mos_model.load_state_dict(ckpt["state_dict"])
        mos_model = mos_model.cuda()
        mos_model.eval()
        mos_model.freeze()
    # 单独处理背景为一个整体
    if cfg.use_bg:
        bg_objects = {c: None for c in BG_CAPTIONS_Pro_Sim}
        # 加载SBERT模型
        sbert_model = SentenceTransformer(cfg.sbert_path)
        sbert_model = sbert_model.to("cuda")
        # 把这些背景的caption先编码了
        bg_fts = []
        for bg_cation in BG_CAPTIONS:
            bg_ft = sbert_model.encode(bg_cation, convert_to_tensor=True)
            bg_ft = bg_ft / bg_ft.norm(dim=-1, keepdim=True)
            bg_ft = bg_ft.squeeze()
            bg_fts.append(bg_ft)
    else:
        bg_objects = None
    point_clouds = []
    for idx in trange(len(datasets)):
        # 第0帧点云太稀疏了，不要了，并且无法估计动态目标
        if idx == 0:
            continue
        image, pc, pose, his_pcs, his_poses = datasets[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 对历史帧的点云进行累积，顺便滤除动态物体
        if his_pcs is not None:
            pc = accumulate_pc(cfg, mos_model, pc, pose, his_pcs, his_poses)
        # 过滤掉距离值太远的点云，可不用
        if cfg.filter_dis:
            pc = distance_filter(cfg.max_depth, pc)
        # 得到点云投影后的像素值和剩下的点云，是一一对应的
        pro_point, pixels = project(pc, image, datasets.calib)
        file_names = os.path.basename(datasets.color_paths[idx])
        save_path_cap = Path(os.path.join(cfg.save_cap_path, f"cap_{file_names}")).with_suffix(".pkl.gz")
        # 打开保存的文件
        with gzip.open(save_path_cap, "rb") as f:
            gobs = pickle.load(f)
        # 得到当前帧的DetectionList类型的点云
        detection_list, bg_list = gobs_to_detection_list(
            cfg = cfg,
            image = image,
            pc = pro_point,
            pixels = pixels,
            idx = idx,
            gobs = gobs,
            trans_pose = pose,
            bg_fts = bg_fts,
            BG_CAPTIONS_Pro = BG_CAPTIONS_Pro,
        )
        # 先单独处理背景
        if len(bg_list) > 0:
            for detected_object in bg_list:
                class_name = detected_object['bg_class']
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, bg=True, class_name = class_name)
        # 如果没有值得加入全局地图的点云
        if len(detection_list) == 0:
            continue
        # 如果是第一帧，全部加入
        if len(objects) == 0:
            for i in range(len(detection_list)):
                objects.append(detection_list[i])
            # 并且跳过下面的相似度计算
            continue
        # 可视化一下
        if cfg.vis_all:
            point_clouds.extend([detection_list[i]["pcd"] for i in range(len(detection_list))])
        # 计算相似度
        spatial_sim = compute_spatial_similarities(detection_list, objects)
        caption_sim = compute_caption_similarities(detection_list, objects)
        ft_sim = compute_ft_similarities(detection_list, objects)
        agg_sim = aggregate_similarities(cfg, spatial_sim, ft_sim, caption_sim)
        # DEBUG: 相似性判断
        # debug_sim = np.dstack((spatial_sim, caption_sim,ft_sim,agg_sim))
        # for i in range(debug_sim.shape[0]):
        #     for j in range(debug_sim.shape[1]):
        #         # 只看有重叠的
        #         if (debug_sim[i][j][0]>0):
        #             print(detection_list[i]["caption"], "***VS***",objects[j]["caption"],debug_sim[i][j])
        # 设置阈值判断是否一个物体。如果低于阈值，则设置为负无穷大
        agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')
        # 按照相似性融合
        objects = merge_detections_to_objects(cfg, detection_list, objects, agg_sim)
    if cfg.vis_all:
        o3d.visualization.draw_geometries(point_clouds)
    
    # 构建完地图之后，降采样地图降分辨率，去噪一下
    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects, bg = True)
    objects = denoise_objects(cfg, objects)
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    # show_captions(objects, bg_objects)
    # 根据最后的结果，融合物体
    objects, generator = caption_merge(cfg, objects)
    # 最后再计算一下融合的caption的特征
    if cfg.caption_merge_ft:
        objects, bg_objects = captions_ft(objects, bg_objects, sbert_model)
    # show_captions(objects, bg_objects)
    # 根据最后的结果，分类得到class
    objects, bg_objects = class_objects(cfg, sbert_model, objects, bg_objects, generator)
    show_captions(objects, bg_objects)

    # 保存一个处理的地图
    if cfg.save_pcd:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'objects': objects.to_serializable(),
            'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': cfg,
        }
        pcd_save_path = cfg.save_pcd_path/f"full_pcd.pkl.gz"
        # 如果目录不存在则创建
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"保存点云地图到 {pcd_save_path}")

if __name__ == "__main__":
    main()        