"""
2024.01.22 
构建路网
"""
import sys
sys.path.append("/home/dyn/outdoor/omm")
import os
import numpy as np
import open3d as o3d
import hydra
from omegaconf import DictConfig
# from some_class.dividing_roadnet import roadnet
from scipy.spatial.transform import Rotation as R
# from visualize import lines_from_ordered_points, create_edge_mesh
from collections import defaultdict





def init_graph(graph_file,height):
    # 读取节点和边数据
    with open(graph_file, 'r') as f:
        lines = f.readlines()
    # 解析节点数据
    nodes = []
    for line in lines:
        line = line.strip()
        if not line:
            break
        parts = line.split()
        x, y,z = float(parts[0]), float(parts[1]),height
        nodes.append((x,y,z))
    # 解析边数据
    edges = []
    for line in lines[len(nodes) + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        u, v,w = int(parts[0]), int(parts[1]),int(parts[2])
        edges.append((u, v,w))
        # print((u, v,w))
    return nodes,edges

def road_point():
    graph_file = os.path.join( './test/05_3D.graph') 
    #读取文件中的点线
    points,lines=init_graph(graph_file,100)
    # print(points)
    # print(lines)
    # lines = np.asarray(okk.edges)
    # 聚类结果字典，用于存储每个权值对应的点
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

    # 输出分类结果
    for weight, points_list in clusters.items():
        print(f"权值 {weight} 对应的点:")
        for point in points_list:
            print(point)
    return clusters
     
def color_by_road_net():
    #point and line

    graph_file = os.path.join( './test/05_3D.graph') 
    
    points,lines_0=init_graph(graph_file,5)
    #读取文件中的点线
    # lines = okk.edges
    lines = np.asarray(lines_0)
    # 创建球体和圆柱体
    sphere_radius = 5  # 球体半径
    cylinder_radius = 2.5  # 圆柱体半径
    # 创建球体和圆柱体列表
    spheres = []
    cylinders = []
    spheres_road=[]
    #上色
    # 路网层圆柱体
    color_cylinder = [0, 1, 0]  
    #路网层球体 道路层四类球体
    color_sphere = [[0,  1 ,0 ],[0, 0 , 0.7],[0.7, 1, 0],[0.7, 0, 1],[0.7, 1, 1]] 
    # 创建已绘制线段的集合
    drawn_lines = set()
    # 创建路网球体
    for point in points:
        # print(point)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color_sphere[0])
        spheres.append(sphere)   
    # 创建圆柱体
    for line in lines:
            idx1, idx2 = line[0], line[1]
            if (idx1, idx2) in drawn_lines or (idx2, idx1) in drawn_lines:
                continue  # 如果线段已经绘制过，则跳过
            drawn_lines.add((idx1, idx2))

            start_point = np.array(points[idx1])
            end_point = np.array(points[idx2])
            direction = end_point - start_point
            length = np.linalg.norm(direction)
            
            direction_normalized = direction / length
            # 计算与方向向量正交的一个向量作为新的 y 轴
            y_axis = np.array([0, 1, 0])
            x_axis = np.cross(direction_normalized, y_axis)
            # 归一化新的 x 轴
            x_axis_normalized = x_axis / np.linalg.norm(x_axis)
            # 计算新的 y 轴
            y_axis_normalized = np.cross(direction_normalized, x_axis_normalized)
            # 构建旋转矩阵
            rotation = np.eye(3)
            rotation[:, 0] = x_axis_normalized
            rotation[:, 1] = y_axis_normalized
            rotation[:, 2] = direction_normalized

            # 获取平移向量
            translation = start_point + direction * 0.5
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=length)
            cylinder.rotate(rotation, center=(0, 0, 0))  # 将圆柱体旋转到正确的方向
            cylinder.translate(translation)  # 将圆柱体平移到正确的位置
            # 设置圆柱颜色
            cylinder.paint_uniform_color(color_cylinder)
            cylinders.append(cylinder)
    
    
    # 创建道路层球体
    road_points=road_point()
    for weight, points_list in road_points.items():
        for point in points_list:      
            # print(point)
            sphere_road = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius*2)
            sphere_road.translate(point)
            sphere_road.paint_uniform_color(color_sphere[weight])
            spheres_road.append(sphere_road) 

    #创建环境层球体
    Toppoint=(-4.447468, 173.154769, 200)
    sphere_env = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 2)
    sphere_env.translate(Toppoint)
    sphere_env.paint_uniform_color(color_sphere[1])
    spheres_road.append(sphere_env)

    #创建环境层与道路层的连接线
    lines = []
    for road_sphere in spheres_road:  # 遍历除了环境层球体之外的道路层球体
        lines.append(road_sphere.get_center())
    # print(np.vstack(lines))
    # 将连接线添加到场景中
    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(np.vstack(lines)),
    lines=o3d.utility.Vector2iVector(np.array([[i, len(lines)-1] for i in range(len(lines))]))
     )
    line_set.paint_uniform_color([0, 0, 1])  # 设置连接线的颜色
    
    
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # for sphere in spheres:
    #     vis.add_geometry(sphere)
    # for sphere_road in spheres_road:
    #     vis.add_geometry(sphere_road)
    # for cylinder in cylinders:
    #     vis.add_geometry(cylinder)
    # vis.add_geometry(line_set)
    # # 运行可视化
    # vis.run()
    # vis.destroy_window()
    return spheres,spheres_road,cylinders,line_set

    
    



@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):
    color_by_road_net()
    #road_point()
if __name__ == "__main__":
    main()
