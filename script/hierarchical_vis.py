import sys
sys.path.append("/home/dyn/outdoor/omm")
import open3d as o3d
import hydra
from omegaconf import DictConfig
import numpy as np
import json
import pickle
import gzip
import distinctipy
from some_class.map_calss import MapObjectList
from tqdm import trange
import copy
import random  # 导入随机模块
from script.roadnet_xzy import color_by_road_net

def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

def create_edge_lines(points, lines=None, colors=[0, 1, 0]):
    '''
    用于以直线形式可视化scenegraph中object的连接边
    '''
    edge_points = np.array(points)
    edge_lines = np.array(
        lines) if lines is not None else lines_from_ordered_points(edge_points)
    edge_colors = np.array(colors)    
    # 创建直线
    edge_line_set = o3d.geometry.LineSet()
    edge_line_set.points = o3d.utility.Vector3dVector(edge_points)
    edge_line_set.lines = o3d.utility.Vector2iVector(edge_lines)

    if edge_colors.ndim == 1:
        edge_line_set.colors = o3d.utility.Vector3dVector(np.tile(edge_colors, (len(edge_lines), 1)))
    else:
        edge_line_set.colors = o3d.utility.Vector3dVector(edge_colors)

    return edge_line_set

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
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
    else:
        raise ValueError("Unknown results type: ", type(results))
    return objects, bg_objects

def load_colors(cfg):
    '''
    类别语义颜色文件
    '''
    file_path = cfg.class_colors_json
    with open(file_path, 'r') as json_file:
        class_colors_sk_disk = json.load(json_file)
    class_names_sk = list(class_colors_sk_disk.keys())
    class_colors_sk = list(class_colors_sk_disk.values())
    class_colors_sk = np.array(class_colors_sk)
    return class_colors_sk_disk, class_names_sk, class_colors_sk


@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    Instance_trans = np.array([0, -90, 0])
    # our_pcd = o3d.io.read_point_cloud("../results/05/pcd/rgb_pc.pcd") # 读取pcd文件
    our_pcd = o3d.io.read_point_cloud("../results/05/pcd/our_pc.pcd") # 读取pcd文件
    
    # 加载pcd结果
    objects, bg_objects = load_result("../results/05/pcd/full_pcd_llama_all.pkl.gz")
    # 不用背景物体
    # if bg_objects is not None:
    #     indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
    #     objects.extend(bg_objects)
    
    bboxes = copy.deepcopy(objects.get_values("bbox"))

    #从script.roadnet_xzy中取到路网层球体、路网层连接圆柱、道路层球体（其中最后一个是最上边环境层球体）、道路层和环境层连线 
    spheres,cylinders,spheres_road,line_set,spheres_colors=color_by_road_net()
    # 获得道路层球体中心的xy值
    spheres_centers = []

    for i in range(len(spheres_road) - 1):
        spheres_center = spheres_road[i].get_center()
        spheres_centers.append(spheres_center)
        
    spheres_centers = np.array(spheres_centers)
    spheres_xy = spheres_centers[:, [0, 2]]
    # 扩展维数以便后续计算
    spheres_xy = np.expand_dims(spheres_xy, axis=0)
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'Open3D', width=1920, height=1280)
    view_control = vis.get_view_control()
    
    import os
    loaded_view_params = None
    if os.path.isfile("/home/dyn/outdoor/omm/test/param.json"):
        loaded_view_params = o3d.io.read_pinhole_camera_parameters("/home/dyn/outdoor/omm/test/param.json")
        view_control.convert_from_pinhole_camera_parameters(loaded_view_params)
    
    vis.add_geometry(our_pcd.voxel_down_sample(1))
    
    
    # 加载物体间关系
    scene_graph_geometries = []
    with open("../results/05/pcd/object_relations_all.json", "r") as f:
        edges = json.load(f)
    # 最好展示的物体都有边连接
    pair_indices_set = set()
    for edge in edges:
        id1 = edge["object1"]["id"]
        id2 = edge["object2"]["id"]
        pair_indices_set.add((id1, id2))
    pair_indices_set = list(pair_indices_set)
    pair_indices_set = np.array(pair_indices_set)
    
    # 采样200对关系，具体多少物体不得而知
    num_sampled_pairs = min(300, pair_indices_set.shape[0])  
    sampled_indices = random.sample(range(pair_indices_set.shape[0]), num_sampled_pairs)  # 随机选择索引
    sampled_pairs = pair_indices_set[sampled_indices]
    # 把涉及到的索引提取出来
    # 创建一个空集合
    my_set = set()
    # 遍历矩阵的每个元素
    for i in range(num_sampled_pairs):
        my_set.add(sampled_pairs[i,0])
        my_set.add(sampled_pairs[i,1])
    sampled_indices = list(my_set)
   
    inst_sphere_lines = []
    instances = []
    for i in sampled_indices:
        bbox_center = bboxes[i].get_center()
        # 创建灰色正方体
        cube = o3d.geometry.TriangleMesh.create_box(width=1.5, height=2.5, depth=1.5)
        # 平移到bbox_center位置
        cube.translate(bbox_center)
        cube.translate(Instance_trans)      

        bbox_xy = np.array([[bbox_center[0], bbox_center[2]]])
        bbox_xy = np.expand_dims(bbox_xy, axis=1)    
        # 计算当前小方块和道路层各个球体的距离    
        distances = np.linalg.norm(spheres_xy - bbox_xy, axis=2)
        # 得到最近距离的索引
        closest_sphere_index = np.argmin(distances)

        # 添加连接边
        inst_sphere_line = create_edge_lines(
                points = np.array([bbox_center + Instance_trans, spheres_centers[closest_sphere_index]]),
                lines = np.array([[0, 1]]),
                colors = spheres_colors[closest_sphere_index],            
        )
        inst_sphere_lines.append(inst_sphere_line)
        
        # 设置颜色
        gray_color = spheres_colors[closest_sphere_index]
        cube.paint_uniform_color(gray_color)
        cube.compute_vertex_normals()
        instances.append(cube)
        
    obj_centers = []
    for obj in objects:
        bbox = obj['bbox']
        obj_center = np.array(bbox.get_center())
        # 将物体中心移到实例层
        obj_center += Instance_trans
        obj_centers.append(obj_center)
    unique_indices = set()
    for edge in edges:
        id1 = edge["object1"]["id"]
        id2 = edge["object2"]["id"]
        unique_indices.add(id1)
        unique_indices.add(id2)
        if id1 in sampled_indices and id2 in sampled_indices:
            edge_line = create_edge_lines(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [0.2, 0.2, 0.2],
            )
            scene_graph_geometries.append(edge_line)

    def save_view_params(vis):
        loaded_view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("/home/dyn/outdoor/omm/test/param.json", loaded_view_params)

    def restore_viewpoint(vis):
        if os.path.isfile("/home/dyn/outdoor/omm/test/param.json"):
            print("view_control ing")
            loaded_view_params = o3d.io.read_pinhole_camera_parameters("/home/dyn/outdoor/omm/test/param.json")
            view_control.convert_from_pinhole_camera_parameters(loaded_view_params)
            print("view_control ed")
            

    main.show_instance = False
    def vis_instance(vis):
        if main.show_instance:
            for geometry in instances:
                # 将正方体添加到可视化窗口
                vis.remove_geometry(geometry)
        else:
            for geometry in instances:
                vis.add_geometry(geometry, reset_bounding_box=False)
        if loaded_view_params is not None:
            view_control.convert_from_pinhole_camera_parameters(loaded_view_params)
        print("main.show_instance", main.show_instance)
        main.show_instance = not main.show_instance   

    main.show_scene_graph = False
    def vis_scene_graph(vis):
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        if loaded_view_params is not None:
            view_control.convert_from_pinhole_camera_parameters(loaded_view_params)
        print("main.show_scene_graph", main.show_scene_graph)
        main.show_scene_graph = not main.show_scene_graph         
    
    main.show_layer234 = False
    def show_layer245_vis(vis):
        if main.show_layer234:
            for sphere in spheres:
                vis.remove_geometry(sphere)
            for sphere_road in spheres_road:
                vis.remove_geometry(sphere_road)
            for cylinder in cylinders:
                vis.remove_geometry(cylinder)
            
        else:
            for sphere in spheres:
                vis.add_geometry(sphere)
            for sphere_road in spheres_road:
                vis.add_geometry(sphere_road)
            for cylinder in cylinders:
                vis.add_geometry(cylinder)
            
        if loaded_view_params is not None:
            view_control.convert_from_pinhole_camera_parameters(loaded_view_params)
        print("main.show_layer234", main.show_layer234)
        main.show_layer234 = not main.show_layer234 
    
    main.show_inst_sphere_lines = False
    def vis_inst_sphere(vis):
        if main.show_inst_sphere_lines:
            for line in inst_sphere_lines:
                vis.remove_geometry(line, reset_bounding_box=False)
            vis.remove_geometry(line_set)
        else:
            for line in inst_sphere_lines:
                vis.add_geometry(line, reset_bounding_box=False)
            vis.add_geometry(line_set)
        if loaded_view_params is not None:
            view_control.convert_from_pinhole_camera_parameters(loaded_view_params)
        print("main.show_inst_sphere_lines", main.show_inst_sphere_lines)
        main.show_inst_sphere_lines = not main.show_inst_sphere_lines             


    vis.register_key_callback(ord("V"), save_view_params)  
    vis.register_key_callback(ord("X"), restore_viewpoint)  
    vis.register_key_callback(ord("I"), vis_instance)
    vis.register_key_callback(ord("G"), vis_scene_graph)
    vis.register_key_callback(ord("O"), show_layer245_vis)       
    vis.register_key_callback(ord("L"), vis_inst_sphere)
    vis.run()
    

if __name__ == "__main__":
    main()    