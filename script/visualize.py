"""
2024.01.22 
将得到的地图可视化并实现查询等
"""
import sys
sys.path.append("/code1/dyn/github_repos/OpenGraph")
import copy
import networkx as nx
from scipy.spatial import KDTree
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
import open_clip
import hydra
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
import distinctipy
from some_class.map_calss import MapObjectList
from utils.utils import  load_models
import json
from PIL import Image
from pathlib import Path
from tqdm import trange
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from some_class.datasets_class import SemanticKittiDataset
import re
import openai

saved_viewpoint = None

###### vis utils defination #####
def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    用于以color ball的方式可视化每个object的中心位置
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def create_edge_mesh(points, lines=None, colors=[0, 1, 0], radius=0.15):
    """
    用于以line mesh的方式可视化scenegraph中object的场景连接边
    """
    edge_points = np.array(points)
    edge_lines = np.array(
        lines) if lines is not None else lines_from_ordered_points(edge_points)
    edge_colors = np.array(colors)
    edge_radius = radius
    edge_cylinder_segments = []

    first_points = edge_points[edge_lines[:, 0], :]
    second_points = edge_points[edge_lines[:, 1], :]
    line_segments = second_points - first_points
    line_segments_unit, line_lengths = normalized(line_segments)

    z_axis = np.array([0, 0, 1])
    # Create triangular mesh cylinder segments of line
    for i in range(line_segments_unit.shape[0]):
        line_segment = line_segments_unit[i, :]
        line_length = line_lengths[i]
        # get axis angle rotation to allign cylinder with line segment
        axis, angle = align_vector_to_another(z_axis, line_segment)
        # Get translation vector
        translation = first_points[i, :] + line_segment * line_length * 0.5
        # create cylinder and apply transformations
        cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
            edge_radius, line_length)
        cylinder_segment = cylinder_segment.translate(
            translation, relative=False)
        if axis is not None:
            axis_a = axis * angle
            # cylinder_segment = cylinder_segment.rotate(
            #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            # cylinder_segment = cylinder_segment.rotate(
            #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            cylinder_segment = cylinder_segment.rotate(
                R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
        # color cylinder
        color = edge_colors if edge_colors.ndim == 1 else edge_colors[i, :]
        cylinder_segment.paint_uniform_color(color)

        edge_cylinder_segments.append(cylinder_segment)
    return edge_cylinder_segments


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
        # print(instance_colors)
        # instance_colors = results['instance_colors']
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
        instance_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)  # 生成一组视觉上相异的颜色
        instance_colors = {str(i): c for i, c in enumerate(instance_colors)}
        # print(instance_colors)
    else:
        raise ValueError("Unknown results type: ", type(results))
    return objects, bg_objects, instance_colors


@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    # mask_generator = load_models(cfg)
    # result_path = args.result_path
    assert not (cfg.result_path is None), \
        "Either result_path must be provided."
    
    # 加载pcd结果
    objects, bg_objects, instance_colors = load_result(cfg.result_path)
    # 必须要有sbert模型哦
    if not cfg.no_sbert:
        print("Initializing SBERT model...")
        sbert_model = SentenceTransformer(cfg.sbert_path)
        sbert_model = sbert_model.to("cuda")
        print("Done initializing SBERT model.")
    # 必须有tap模型
    if not cfg.no_tap:
        print("Initializing TAP model...")
        model_type = "tap_vit_l"
        checkpoint = cfg.tap_path 
        tap_model = model_registry[model_type](checkpoint=checkpoint)
        concept_weights = cfg.tap_merge_path
        tap_model.concept_projector.reset_weights(concept_weights)
        tap_model.text_decoder.reset_cache(max_batch_size=1000)
        tap_model = tap_model.to("cuda")
        print("Done initializing TAP model.")        
    # 为了生成颜色
    cmap = matplotlib.colormaps.get_cmap("turbo")
    # 如果存在背景物体，放在后面一部分
    if bg_objects is not None:
        indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        objects.extend(bg_objects)
    # 下采样点云
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(cfg.voxel_size)
        objects[i]['pcd'] = pcd
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    # gpt加载
    openai.api_key = cfg.openai_key
    openai.api_base = cfg.api_base
    TIMEOUT = 25  # timeout in seconds
    DEFAULT_PROMPT = """
    You are an object picker that picks the three objects from a sequence of objects 
    that are most relevant to the query statement, the first being the most relevant. 
    The input format is: object index value and corresponding description, and query statement. 
    The output format is: The three most relevant objects are [Index0,Index1,Index2].
    Here's some example for you. 
    Input: 
    'The sequence of objects: [0]A silver car.  [1]A green tree.  [2]A paved city road.  [3]A red car. 
    Query statement: driving tools
    '. 
    You should output like this: 
    '[0,3,2]
    '
    Input: 
    'The sequence of objects: [0]A green tree.  [1]A white building.  [2]A paved city road.  [3]A red car. 
    Query statement: a comfortable bed
    '. 
    You should output like this: 
    '[1,3,2]
    '
    Please note! Be sure to give three index values! No more and no less!
    Please note! Be sure to give three index values! No more and no less!
    Please note! Be sure to give three index values! No more and no less!
    
    
    Note that the following are real inputs:
    """
    formatted_sentences = []
    for i in range(len(objects)):
        formatted_sentences.append(f"[{i}]{objects[i]['caption']}.  ")
    caption_all = " ".join(formatted_sentences)
    caption_all = "The sequence of objects: "+caption_all+"\n"
    # print(caption_all)
    
    # 创建可视化对象 设置窗口名
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if cfg.result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(cfg.result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)
    view_control = vis.get_view_control()
    # 加载视角参数
    if os.path.isfile(cfg.vis_sequence+"_vis_params.json"):
        global saved_viewpoint
        loaded_view_params = o3d.io.read_pinhole_camera_parameters(cfg.vis_sequence+"_vis_params.json")
        saved_viewpoint = loaded_view_params
    # 向场景中添加几何图形并创建相应的着色器
    for geometry in pcds:
    # for geometry in pcds + bboxes:
        vis.add_geometry(geometry)  
    # 是否存在场景图配置 以及 场景图的相关可视化语句
    if cfg.scenegraph_vis:
        assert not (cfg.scenegraph_path is None), \
            "Either scenegraph_path must be provided."
        # Load edge files and create meshes for the scene graph
        scene_graph_geometries = []
        with open(cfg.scenegraph_path, "r") as f:
            edges = json.load(f)
        # 可视化每个objects的中心位置为ball
        obj_centers = []
        for obj in objects:
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            radius = extent ** 0.5 / 25
            obj_centers.append(center)
            ball = create_ball_mesh(center, radius, obj['inst_color'])
            scene_graph_geometries.append(ball)
        for edge in edges:
            id1 = edge["object1"]["id"]
            id2 = edge["object2"]["id"]
            line_mesh = create_edge_mesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.05
            )
            scene_graph_geometries.extend(line_mesh)


    main.show_bg_pcd = True
    def toggle_bg_pcd(vis):
        '''
        隐藏or显示背景物体
        '''
        if bg_objects is None:
            print("No background objects found.")
            return
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        main.show_bg_pcd = not main.show_bg_pcd

    def color_by_instance(vis):
        '''
        根据实例随机着色
        '''
        # 设置每个对象的点云颜色
        for i in range(len(objects)):
            color = instance_colors[str(i)]
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_class(vis):
        '''
        根据语义类别，按照semantickitti类别着色
        '''
        # 设置每个对象的点云颜色
        for i in range(len(objects)):
            color = objects[i]["inst_color"]
            # print(color)
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_rgb(vis):
        '''
        RGB着色
        '''
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        for pcd in pcds:
            vis.update_geometry(pcd)

    def sim_and_update(similarities, vis):
        '''
        文本、图像查询的相似性计算及点云更新
        '''
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)      
        max_prob_idx = torch.argmax(probs)       
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]        
        # 更新点云对象的颜色属性，以反映每个对象的相似性
        for i in range(len(objects)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(        
                np.tile(
                    [
                        similarity_colors[i, 0].item(),     
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)    
                )
            )
        for pcd in pcds:
            vis.update_geometry(pcd)        

    def color_by_sbert_sim(vis):
        '''
        文本查询
        '''
        text_query = input("Enter your query: ")
        text_queries = [text_query]
        # 对输入的文本进行编码
        text_query_ft = sbert_model.encode(text_queries, convert_to_tensor=True)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        objects_sbert_fts = objects.get_stacked_values_torch("ft")
        objects_sbert_fts = objects_sbert_fts.to("cuda")
        if cfg.caption_merge_ft:
            # 物体的特征，来自融合后的caption现在编码的特征和增量融合的特征
            objects_caption_fts = objects.get_stacked_values_torch("captions_ft")
            objects_sbert_fts = objects_sbert_fts*cfg.vis_ft_weight + objects_caption_fts*cfg.vis_caption_weight
        similarities = F.cosine_similarity(text_query_ft.unsqueeze(0), objects_sbert_fts, dim=-1)
        top_values, top_indices = similarities.topk(3)
        print(top_indices)
        print("The TOP3 captions are")
        print(objects[top_indices[0]]["caption"])
        print(objects[top_indices[1]]["caption"])
        print(objects[top_indices[2]]["caption"])
        sim_and_update(similarities, vis)

    
    def color_by_llm(vis):
        '''
        文本查询
        '''
        text_query = input("Enter your query for LLM: ")
        text_queries = "Query statement: "+text_query
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + caption_all+text_queries}],
            timeout=TIMEOUT,  # Timeout in seconds
        )
        input_text = chat_completion["choices"][0]["message"]["content"]
        print(input_text)
        pattern = r'\[([^]]+)\]'  # 匹配方括号中的内容
        match = re.search(pattern, input_text)
        extracted_content = []
        if match:
            extracted_content = match.group(1)
        index = [int(x) for x in extracted_content.split(',')]
        print(index)
        # 更新点云对象的颜色属性，以反映每个对象的相似性
        for i in range(len(objects)):
            if i == index[0]:
                pcd = pcds[i]
                print(objects[i]["caption"])
                pcd.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(pcd.points), 1)))
            elif i == index[1]:
                pcd = pcds[i]
                print(objects[i]["caption"])
                pcd.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(pcd.points), 1)))
            elif i == index[2]:
                pcd = pcds[i]
                print(objects[i]["caption"])
                pcd.colors = o3d.utility.Vector3dVector(np.tile([0,0,1], (len(pcd.points), 1)))
            else:
                pcd = pcds[i]
                pcd.colors = o3d.utility.Vector3dVector(np.tile([0.7,0.7,0.7], (len(pcd.points), 1)))
        for pcd in pcds:
            vis.update_geometry(pcd)        

        

    def show_captions(mask, caption, ax, original_size):
        '''
        保存图像可视化用
        '''
        true_coords = np.argwhere(mask)
        if len(true_coords) > 0:
            color = np.concatenate([np.random.random(3), np.array([1])], axis=0)
            ax.imshow(mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1), alpha=0.9, label=f'Mask {i+1}')
            box = 0 , 0 , original_size[1] , original_size[0]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=0.5, edgecolor='r', facecolor='none')
            # rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=0.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            center_x = (box[0]+box[2])/2
            center_y = (box[1]+box[3])/2
            print(center_x,center_y)
            caption_width = len(caption) * 5  # 根据字体大小调整
            caption_x = center_x - caption_width / 2
            caption_y = center_y
            ax.text(caption_x, caption_y, f"{caption}", color='black', fontsize=8, bbox=dict(facecolor=color, alpha=1.0))

 
    def color_by_image_sim(vis):
        '''
        图像查询
        '''
        # 输入图像
        image_query = input("Enter the picture name: ")
        image_base_path = '/code1/dyn/github_repos/OpenGraph/image_query/'
        input_image_path = '{}{}'.format(image_base_path, image_query)
        input_image = cv2.imread(input_image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # input_image = np.array(cv2.imread(input_image_path))
        # input_image = np.array(Image.open(input_image_path))
        # print(input_image)
        # 图像预处理
        vis_img = input_image.copy()[:, :, ::-1]
        img_list, img_scales = im_rescale(input_image, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, input_image.shape[:2]
        img_batch = im_vstack(img_list, fill_value=tap_model.pixel_mean_value, size=(1024, 1024))
        inputs = tap_model.get_inputs({"img": img_batch})
        inputs.update(tap_model.get_features(inputs))
        # 存储边界框信息
        batch_points = np.zeros((1, 2, 3), dtype=np.float32)
        batch_points[0, 0, :2] = 1, 1
        batch_points[0, 1, :2] = original_size[1]-2, original_size[0]-2
        batch_points[0, 0, 2] = 2
        batch_points[0, 1, 2] = 3
        # print(batch_points)
        inputs["points"] = batch_points
        inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")
        # 开始推理，得到mask大小
        outputs = tap_model.get_outputs(inputs)
        iou_pred = outputs["iou_pred"].detach().cpu().numpy()
        point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
        rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
        mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
        mask_pred = outputs["mask_pred"]
        masks = mask_pred[mask_index]
        masks = tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = tap_model.upscale_masks(masks, original_size).gt(0).cpu().numpy()
        # 推理captions
        sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
        captions = tap_model.generate_text(sem_tokens)
        print(captions)
        caption_fts = sbert_model.encode(captions, convert_to_tensor=True, device="cuda")
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        # 保存caption结果到图像中
        if cfg.save_image_vis:
            plt.figure(figsize=(20,8))
            plt.imshow(vis_img)
            show_captions(masks, captions, plt.gca(), original_size)
            plt.axis('off')
            # 如果提供了保存路径，则保存图像
            if cfg.save_image_vis_path:
                Path(cfg.save_image_vis_path).parent.mkdir(parents=True, exist_ok=True)
                random_file_name = str(random.randint(1,100))
                save_image_vis_path = Path(os.path.join(cfg.save_image_vis_path, f"vis_{random_file_name}"))                
                plt.savefig(str(save_image_vis_path), bbox_inches='tight', pad_inches=0)
                plt.close() 
        # 物体的特征，来自融合后的caption现在编码的特征和增量融合的特征
        objects_sbert_fts = objects.get_stacked_values_torch("ft")
        objects_sbert_fts = objects_sbert_fts.to("cuda")
        if cfg.caption_merge_ft:
            # 物体的特征，来自融合后的caption现在编码的特征和增量融合的特征
            objects_caption_fts = objects.get_stacked_values_torch("captions_ft")
            objects_sbert_fts = objects_sbert_fts*cfg.vis_ft_weight + objects_caption_fts*cfg.vis_caption_weight
        similarities = F.cosine_similarity(caption_fts, objects_sbert_fts, dim=-1)    
        sim_and_update(similarities, vis)
        
    main.show_scene_graph = False
    def vis_scene_graph(vis):
        if cfg.scenegraph_path is None or (not cfg.scenegraph_vis):
            print("No scenegraph file provided or scenegraph not supported.")
            return
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        main.show_scene_graph = not main.show_scene_graph
    
    def save_viewpoint(vis):
        global saved_viewpoint
        saved_viewpoint = view_control.convert_to_pinhole_camera_parameters()
        # 保存视角参数
        view_params = view_control.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(cfg.vis_sequence+"_vis_params.json", view_params)
        
    def restore_viewpoint(vis):
        global saved_viewpoint
        if saved_viewpoint is not None:
            view_control.convert_from_pinhole_camera_parameters(saved_viewpoint)
            
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_sbert_sim)
    vis.register_key_callback(ord("P"), color_by_image_sim)  
    vis.register_key_callback(ord("G"), vis_scene_graph)  
    vis.register_key_callback(ord("M"), color_by_llm)   
    vis.register_key_callback(ord("V"), save_viewpoint) 
    vis.register_key_callback(ord("X"), restore_viewpoint)       
    vis.run()   

if __name__ == "__main__":
    main()