"""
2024.02.27 
测试：二维轨迹得到分割后的路网，同时生成道路节点
"""
import sys
import numpy as np
sys.path.append("/home/dyn/outdoor/omm")
import matplotlib.pyplot as plt
import json
from pathlib import Path
from omegaconf import DictConfig
import hydra
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import itertools
from scipy.spatial import distance
from collections import defaultdict
import math
import json



def process_cfg(cfg: DictConfig):
    '''
    配置文件预处理
    '''
    cfg.basedir = Path(cfg.basedir)
    cfg.save_vis_path = Path(cfg.save_vis_path)
    cfg.save_cap_path = Path(cfg.save_cap_path)
    cfg.save_pcd_path = Path(cfg.save_pcd_path)
    return cfg

def calculate_angle(vectors):
    '''
    向量组两两之间夹角
    '''
    # 去除零向量
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    non_zero_indices = np.where(norms != 0)[0]
    vectors = vectors[non_zero_indices]
    for i in range(len(vectors)):
        norm_i = np.linalg.norm(vectors[i])
        vectors[i] = vectors[i] / norm_i  
    angles = []
    raw_angles = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            dot_product = np.dot(vectors[i], vectors[j])
            if not np.isnan(dot_product):
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 修正角度范围
                raw_angles.append(angle)
                angle = min(abs(angle-0),abs(angle-np.pi))
                angles.append(angle)
    mean_angle = np.mean(angles)
    return mean_angle

def process_indexes(indexes):
    '''
    处理列表，保留连续索引中的一个
    '''
    processed_indexes = []
    start_index = None
    for i in range(len(indexes)):
        if start_index is None:
            start_index = indexes[i]
        elif indexes[i] - indexes[i-1] > 1:  # 如果当前索引与前一个索引不连续
            middle_index = (indexes[i-1] + start_index) // 2  # 计算连续索引范围内的中间值
            processed_indexes.append(middle_index)
            start_index = indexes[i]
    # 处理最后一个连续索引范围
    if start_index is not None:
        end_index = indexes[-1]
        middle_index = (end_index + start_index) // 2
        processed_indexes.append(middle_index)
    return processed_indexes

def generate_paths(node_sequence, all_path_nodes):
    '''
    为相邻节点之间生成路径
    '''
    paths = set()
    paths_index = set()
    for i in range(len(node_sequence) - 1):
        start_node = tuple(all_path_nodes[node_sequence[i]])
        end_node = tuple(all_path_nodes[node_sequence[i + 1]])
        start_index = node_sequence[i]
        end_index = node_sequence[i+1]
        paths.add((start_node, end_node))
        paths_index.add((start_index,end_index))
        paths_index.add((end_index,start_index))
    paths = list(paths)
    paths = np.array(paths)
    paths_index = list(paths_index)
    # paths_index = np.array(paths_index)
    return paths, paths_index

def refine_nodes(all_path_nodes, paths_index, dis=15):
    '''
    插入一小块作为管辖区域
    '''
    num = len(all_path_nodes)
    for i in range(num):
        paths_index_array = np.array(paths_index)
        neighbors = np.where(paths_index_array[:, 0] == i)[0]  # 查找与当前节点相连的节点索引
        if len(neighbors) >= 2:  # 如果至少有两个相连节点
            for j in range(len(neighbors)):
                paths_index.remove((i,paths_index_array[neighbors[j], 1]))
                paths_index.remove((paths_index_array[neighbors[j], 1],i))
                start_node = all_path_nodes[i]
                end_node = all_path_nodes[paths_index_array[neighbors[j], 1]]
                # 计算到相邻节点的距离
                dist = np.linalg.norm(start_node - end_node)
                # 计算新节点的位置
                new_node = start_node + (end_node - start_node) * (dis / dist)
                # 加入当前路网
                all_path_nodes = np.vstack((all_path_nodes,new_node))
                paths_index.append((len(all_path_nodes)-1, i))
                paths_index.append((i, len(all_path_nodes)-1))
                paths_index.append((len(all_path_nodes)-1, paths_index_array[neighbors[j], 1]))
                paths_index.append((paths_index_array[neighbors[j], 1], len(all_path_nodes)-1))
    paths_index = np.array(paths_index)
    return all_path_nodes, paths_index

def get_other_vertex(u, v, vertex_connections):
    # 获取顶点u与v之外的另一个相连顶点
    for vertex in vertex_connections[u]:
        if vertex != v:
            return vertex
    for vertex in vertex_connections[v]:
        if vertex != u:
            return vertex

#计算三点组成边的夹角，[0,180度]
def calculate_angle_point(point1, point2, point3):
    # 计算向量1
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    # 计算向量2
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    # 计算向量1和向量2的点积
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    # 计算向量1和向量2的模
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    num = dot_product / (magnitude1 * magnitude2)
    # print(num)
    if (num)>1:
        num = 1
    if (num)<-1:
        num = -1
    # 计算夹角（弧度）
    angle_rad = math.acos(num)
    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)
    return angle_deg
    
def dividing(nodes, edges):
    '''
    分割路网
    '''
    # 统计每个顶点连接的其他顶点数量
    vertex_connections = defaultdict(set)
    for edge in edges:
        u, v, _ = edge
        vertex_connections[u].add(v)
        vertex_connections[v].add(u)
    # 根据连接的顶点数量设置权重
    for i, edge in enumerate(edges):
        u, v, _ = edge
        connections_max = max(len(vertex_connections[u]), len(vertex_connections[v]))
        if connections_max == 2:
            other_vertex = get_other_vertex(u, v, vertex_connections)
            x = [nodes[u][0], nodes[u][1]]
            y = [nodes[v][0], nodes[v][1]]
            z=  [nodes[other_vertex][0], nodes[other_vertex][1]]
            angle = calculate_angle_point(x,y,z)
            if angle < 50 and angle > 10:
                edges[i] = (u, v, 2)
        elif connections_max == 3:
            edges[i] = (u, v, 3)
        elif connections_max == 4:
            edges[i] = (u, v, 4)               
        else:
            edges[i] = (u, v, 1)
        # 使一条边的w相同
        for i, edge in enumerate(edges):
            u, v, _ = edge
            if (v, u, 2) in edges and (u, v, 1) in edges:
                edges[i] = (u, v, 2)
    return edges
                
                
def road_point(points, lines):
    clusters = {
        1: [],
        2: [],
        3: [],
        4: []
    }
    vertex_2_count = defaultdict(int)
    vertex_3_count = defaultdict(int)
    vertex_4_count = defaultdict(int)
    # 对每个路段进行处理
    for line in lines:
        start_idx, end_idx, weight = line
        start_point = points[start_idx]
        end_point = points[end_idx]
        # 对于权值为1的情况，将当前路段中的点添加到第一类点中
        if weight == 1:
            midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2,(start_point[2] + end_point[2]) / 2)
            if midpoint not in clusters[1]:
                clusters[1].append(midpoint)
        # 对于权值为2的情况，计数       
        elif weight == 2:
            vertex_2_count[start_idx] += 1
            vertex_2_count[end_idx] += 1
        elif weight == 3:
            vertex_3_count[start_idx] += 1
            vertex_3_count[end_idx] += 1
        elif weight == 4:
            vertex_4_count[start_idx] += 1
            vertex_4_count[end_idx] += 1
    # 从计数器中选择出现四次的顶点放入第2、3、4类
    clusters[2] = [points[vertex] for vertex, count in vertex_2_count.items() if count == 4]
    clusters[3] = [points[vertex] for vertex, count in vertex_3_count.items() if count == 6]
    clusters[4] = [points[vertex] for vertex, count in vertex_4_count.items() if count == 8]
    return clusters      
                

@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    pose_save_path = "/home/dyn/outdoor/results/05/trans_poses.txt"
    current_poses = []
    with open(pose_save_path, 'r') as f:
        for line in f:
            current_poses.append(json.loads(line.strip()))
    current_poses = np.array(current_poses)       
    # 使用KD树构建二维轨迹点的索引
    kd_tree = KDTree(current_poses)

    # 计算每个轨迹点的夹角差值
    mean_angles = []
    duandians = []
    for current_pose in current_poses:
        x, y = current_pose
        # 使用KD树查询半径20m内的所有点
        near_indices = kd_tree.query_radius(np.array([x, y]).reshape(1, -1), r=20)[0]
        near_points = current_poses[near_indices]
        vectors = [np.array(near_point) - np.array(current_pose) for near_point in near_points]
        mean_angle = calculate_angle(vectors)
        mean_angles.append(mean_angle)
        # 断点是太少的临界点
        if near_points.shape[0] < 3:
            duandians.append(current_pose)
    mean_angles=np.array(mean_angles)
    # print(duandians)
    # 使用DBSCAN进行聚类
    mask = mean_angles > 0.15
    use_points = current_poses[mask]
    dbscan = DBSCAN(eps=20, min_samples=3)
    cluster_labels = dbscan.fit_predict(use_points)

    # 找到每个簇的中心点
    unique_labels = np.unique(cluster_labels)
    cluster_centers = []
    for label in unique_labels:
        if label == -1:
            continue  # -1表示噪声点
        cluster_points = use_points[cluster_labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
        
    # 保留每个簇中心点
    unique_cluster_centers = np.array(cluster_centers)\
    # 所有道路结点
    all_path_nodes = np.vstack((unique_cluster_centers,duandians))
    
    # 计算轨迹点与道路节点之间的距离
    distances = distance.cdist(all_path_nodes, current_poses)
    # 为每个道路节点找到距离小于阈值的轨迹点的索引列表
    indexes_list = []
    for i in range(len(all_path_nodes)):
        indexes = np.where(distances[i] < 20)[0]
        indexes_list.append(indexes)
    # for i, indexes in enumerate(indexes_list):
    #     print(f"道路节点 {i} 的轨迹点索引列表：{indexes}")
    # 处理索引列表，连续只保留一个
    processed_indexes_list = [process_indexes(indexes) for indexes in indexes_list]
    # for i, indexes in enumerate(processed_indexes_list):
    #     print(f"道路节点 {i} 的处理后的轨迹点索引列表：{indexes}")

    # 将处理后的列表中的节点索引映射到节点编号
    node_index_mapping = {}
    for node_index, indexes in enumerate(processed_indexes_list):
        for index in indexes:
            node_index_mapping[index] = node_index
    node_sequence = []
    for index in sorted(node_index_mapping.keys()):
        node_sequence.append(node_index_mapping[index])
    print("道路节点顺序：", node_sequence)
    
    # 得到路径
    paths, paths_index = generate_paths(node_sequence, all_path_nodes)
    
    nodes, edges = refine_nodes(all_path_nodes, paths_index, dis=20)
    
    # 新增加的列
    new_column = np.ones((edges.shape[0], 1))  # 创建一个 n×1 的全零数组
    # 在原始数组中添加新列
    edges = np.hstack((edges, new_column)).astype(int)
    
    new_column = np.zeros((nodes.shape[0], 1))  # 创建一个 n×1 的全零数组
    # 在原始数组中添加新列
    nodes = np.hstack((nodes, new_column))
    
    nodes = [tuple(row) for row in nodes]
    edges = [tuple(row) for row in edges]
    edges = dividing(nodes, edges)
    road_points = road_point(nodes, edges)
    
    name = "test/graph+road.json"
    edges = [(int(u), int(v), int(w)) for u, v, w in edges]  # 将 numpy.int64 转换为 int
    road_points = {key: [point for point in points] for key, points in road_points.items()}  # 将 numpy 数组转换为 Python 列表
    data = {
        "nodes": nodes,
        "edges": edges,
        "road_points": road_points
    }
    with open(name, "w") as f:
        json.dump(data, f)
    
    # plt.clf()
    # # 接受到路网之后，绘制路网和路径
    # if nodes is not None:
    #     # 绘制边，
    #     for i, (u, v, w) in enumerate(edges):
    #         x = [nodes[u][0], nodes[v][0]]
    #         y = [nodes[u][1], nodes[v][1]]
    #         #L交叉路口
    #         if w == 2:     
    #             plt.plot(x, y, linewidth=1, color='darkblue')
    #         #T交叉路口   
    #         elif w == 3:
    #             plt.plot(x, y, linewidth=1, color='darkred')
    #             #十字交叉路口   
    #         elif w == 4:
    #             plt.plot(x, y, linewidth=1, color='darkorange')
    #         #正常直路
    #         else:
    #             plt.plot(x, y, linewidth=1, color='darkgreen')
    
    # # 可视化
    # # plt.scatter([p[0] for p in current_poses], [p[1] for p in current_poses], c='black', s=20)
    # # plt.scatter([p[0] for p in current_poses], [p[1] for p in current_poses], c=mean_angles, s=20, cmap='viridis')
    # # plt.scatter([p[0] for p in use_points], [p[1] for p in use_points], c=mean_angles[mask], cmap='viridis')
    # # plt.scatter([p[0] for p in unique_cluster_centers], [p[1] for p in unique_cluster_centers], c='red', s=200, marker='*', label='Cluster Centers')
    # # plt.scatter([p[0] for p in duandians], [p[1] for p in duandians], c='darkorange', s=200, marker='*', label='Cluster Centers')
    # # plt.scatter([p[0] for p in nodes], [p[1] for p in nodes], c='darkorange', s=200, marker='*')
    # # for path in paths:
    # #     plt.plot([path[0][0], path[1][0]], [path[0][1], path[1][1]], c='green')
    # # for edge in edges:
    # #     plt.plot([nodes[edge[0]][0], nodes[edge[1]][0]], [nodes[edge[0]][1], nodes[edge[1]][1]], c='green')
    # # plt.legend()
    # # plt.colorbar(label='Trajectory Order')
    # plt.title("Lane traj Colored by Trajectory Order")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.savefig("traj_colored.png", dpi=500)

if __name__ == "__main__":
    main()
