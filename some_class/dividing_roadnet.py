#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import threading
import os
import time
from collections import defaultdict
import math
from matplotlib.lines import Line2D

class roadnet:
    def __init__(self):
        # 用于不重复接受
        self.net_version = 0 
        # 用于可视化路网
        self.nodes = None
        self.edges = None
        # 用于可视化不同种类的边
        self.delete_edge_1 = None
        self.delete_edge_2 = None
        # 用于可视化路径
        self.path_point = None

    # 初始化一个离线的路网
    def init_graph(self,graph_file):
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
            x, y,z = float(parts[0]), float(parts[1]),5
            nodes.append((x,y,z))
        # 解析边数据
        edges = []
        for line in lines[len(nodes) + 1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u, v,w = int(parts[0]), int(parts[1]),1
            edges.append((u, v,w))
            # print((u, v,w))
        self.nodes = nodes
        self.edges = edges

    #计算三点组成边的夹角，[0,180度]
    def calculate_angle(self,point1, point2, point3):
        # 计算向量1
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        # 计算向量2
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])
        # 计算向量1和向量2的点积
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        # 计算向量1和向量2的模
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        # 计算夹角（弧度）
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        # 将弧度转换为角度
        angle_deg = math.degrees(angle_rad)
        return angle_deg


    def get_other_vertex(self,u, v, vertex_connections):
        # 获取顶点u与v之外的另一个相连顶点
        for vertex in vertex_connections[u]:
            if vertex != v:
                return vertex
        for vertex in vertex_connections[v]:
            if vertex != u:
                return vertex

    def dividing(self):
        # 统计每个顶点连接的其他顶点数量
        vertex_connections = defaultdict(set)
        # print(vertex_connections)
        for edge in self.edges:
            # print(edge)
            u, v, _ = edge
            # print(vertex_connections[u],vertex_connections[v] )
            vertex_connections[u] .add(v)
            vertex_connections[v] .add(u)

        # 根据连接的顶点数量设置权重
        for i, edge in enumerate(self.edges):
            u, v, _ = edge
            connections_max =max(len(vertex_connections[u]), len(vertex_connections[v]))
            # print(connections_max)
            if connections_max == 2:
                other_vertex = self.get_other_vertex(u, v, vertex_connections)
                # print(other_vertex)
                x = [self.nodes[u][0], self.nodes[u][1]]
                y = [self.nodes[v][0], self.nodes[v][1]]
                z=  [self.nodes[other_vertex][0], self.nodes[other_vertex][1]]
                angle = self.calculate_angle(x,y,z)
                # print(angle)
                if angle < 50 and angle >10:
                    self.edges[i] = (u, v, 2)
                # print(self.edges[i])
            elif connections_max == 3:
                self.edges[i] = (u, v, 3)
            elif connections_max == 4:
                self.edges[i] = (u, v, 4)               
            else:
                self.edges[i] = (u, v, 1)
            # print(self.edges[i])

            #使一条边的w相同
            for i, edge in enumerate(self.edges):
                u, v, _ = edge
                if (v, u, 2) in self.edges and (u, v, 1) in self.edges:
                    self.edges[i] = (u, v, 2)

    def save3D(self, graph_file):
     with open(graph_file, 'w') as f:
        # 写入节点数据
        for node in self.nodes:
            x, y, z = node
            f.write(f"{x} {y} {z}\n")
        f.write("\n")
        # 写入边数据
        for edge in self.edges:
            u, v, w = edge
            f.write(f"{u} {v} {w}\n")

    def drawing(self):
        '''绘制图像'''
        # 清除当前图形
        plt.clf()
        # 接受到路网之后，绘制路网和路径
        if self.nodes is not None:
            # 不再绘制结点，因为太多了
            # plt.scatter(*zip(*nodes), color='green', marker='o', label='Road nodes')
            # 绘制边，
            for i, (u, v, w) in enumerate(self.edges):
                x = [self.nodes[u][0], self.nodes[v][0]]
                y = [self.nodes[u][1], self.nodes[v][1]]
                #L交叉路口
                if w == 2:     
                    plt.plot(x, y, linewidth=10, color='darkblue')
                #T交叉路口   
                elif w == 3:
                    plt.plot(x, y, linewidth=10, color='darkred')
                 #十字交叉路口   
                elif w == 4:
                    plt.plot(x, y, linewidth=10, color='darkorange')
                #正常直路
                else:
                    plt.plot(x, y, linewidth=10, color='darkgreen')
            # 设置图像标题和坐标轴标签
            plt.title('Road_Net', fontsize=20)
            plt.xlabel('x')
            plt.ylabel('y')
            # 设置坐标轴比例相同
            plt.axis('equal')
            # 添加图例
            # legend_elements = [
            #     Line2D([0], [0], color='darkblue', lw=5, label='L-intersection'), 
            #     Line2D([0], [0], color='darkred', lw=5, label='T-intersection'),
            #     Line2D([0], [0], color='darkorange', lw=5, label='Crossroad'),
            #     Line2D([0], [0], color='darkgreen', lw=5, label='Straight road')
            # ]
            plt.legend()
            plt.savefig('test/05_2D.png')
            # 刷新图形
            plt.pause(1)
            # plt.show()




if __name__ == '__main__':
    #实例化类
    okk=roadnet()
    #读取路网数据文件
    current_dir = os.path.dirname(__file__)
    graph_file = os.path.join(current_dir,  '05.graph') 
    okk.init_graph(graph_file)
    #分割路网
    okk.dividing()
    #保存路网文件用来3D可视化
    graph_file_3D = "test/05_3D.graph" 
    okk.save3D(graph_file_3D)
    #2D可视化
    okk.drawing()
    
    
    