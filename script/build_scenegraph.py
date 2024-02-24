import sys
sys.path.append("/home/dyn/outdoor/omm")
import pickle
import gzip
from pathlib import Path
from omegaconf import DictConfig
from some_class.map_calss import MapObjectList
import distinctipy
import hydra
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

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

def geometric_dist_matrix(cfg, objects: MapObjectList):
    '''
    用于计算MapObjectList中各个instance之间的距离矩阵--上三角矩阵以简化计算
    '''
    n = len(objects)
    dist_matrix = np.zeros((n, n))

    # Compute the pairwise distances
    for i in range(n):
        for j in range(i,n):
            if i != j:  # Skip diagonal elements
                pcd_i = objects[i]['pcd']
                pcd_i_points = np.asarray(pcd_i.points)
                pcd_i_center = np.mean(pcd_i_points, axis=0)
                pcd_j = objects[j]['pcd']
                pcd_j_points = np.asarray(pcd_j.points)
                pcd_j_center = np.mean(pcd_j_points, axis=0)
                
                # Skip if the boxes do not overlap at all (saves computation)
                distance = np.linalg.norm(pcd_i_center - pcd_j_center)
                
                # Calculate the ratio of points within the threshold
                dist_matrix[i, j] = 1.0 / (distance*100)

    return dist_matrix

def build_SG(cfg, objects: MapObjectList):
    '''
    用于从输入的MapObjectList中构建3D场景图--循循渐进
    1.semantic nodes + geometric edges
    '''
    num_instance = len(objects)
    print("Computing adjacent objects distance...")
    # object_dist has been inversed for distance less weights higher!
    object_dist = geometric_dist_matrix(cfg, objects)

    # Construct a weighted adjacency matrix based on object distances
    weights = []
    rows = []
    cols = []
    for i in range(num_instance):
        for j in range(i + 1, num_instance):
            if i == j:
                continue
            if object_dist[i, j] > 0.0001:
                # restore the weights for objects in the distance of 5m
                # print(f"acctual distance output:{1.0/(bbox_overlaps[i, j]*100)}")
                weights.append(object_dist[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(object_dist[i, j])
                rows.append(j)
                cols.append(i)   
    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_instance, num_instance))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)
    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())
    
    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)
    
        # if not (Path(cfg.save_pcd_path) / "object_relations.json").exists():
        for componentidx, component in enumerate(components):
            if len(component) <= 1:
                continue
            for u, v in zip(
                minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
            ):
                segmentidx1 = component[u]
                segmentidx2 = component[v]
                _bbox1 = objects[segmentidx1]["bbox"]
                _bbox2 = objects[segmentidx2]["bbox"]

                input_dict = {
                    "object1": {
                        "id": segmentidx1,
                        "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                        "bbox_center": np.round(_bbox1.center, 1).tolist(),
                        "object_caption": objects[segmentidx1]["caption"],
                        "object_tag": objects[segmentidx1]["class_sk"],
                    },
                    "object2": {
                        "id": segmentidx2,
                        "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                        "bbox_center": np.round(_bbox2.center, 1).tolist(),
                        "object_caption": objects[segmentidx2]["caption"],
                        "object_tag": objects[segmentidx2]["class_sk"]
                    },
                }

                # TODO: 获得除了空间联系之外的其他的场景边属性 for output_dict

                output_dict = input_dict
                relations.append(output_dict)
        # Saving the output
        print("Saving object relations to file...")
        with open(Path(cfg.save_pcd_path) / "object_relations.json", "w") as f:
            json.dump(relations, f, indent=4)
        # else:
        #     relations = json.load(open(Path(cfg.save_pcd_path) / "object_relations.json", "r"))
    print(f"Created 3D scenegraph with {num_instance} nodes and {len(relations)} edges")
    return relations

@hydra.main(version_base=None, config_path="../config", config_name="semantickitti")
def main(cfg : DictConfig):    
    # 加载pcd结果
    objects, _, _ = load_result(cfg.result_path)
    print("Pcd loaded successfully!")

    # build 3D scene graph
    object_edges = build_SG(cfg, objects)

if __name__ == "__main__":
    main()    