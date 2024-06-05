"""
2024.01.18 
用于点云之间的融合的一些工具代码
"""
import sys
sys.path.append("/code1/dyn/github_repos/OpenGraph")
import torch
import torch.nn.functional as F
from some_class.map_calss import DetectionList, MapObjectList
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import trange
# import for caption merging
from typing import List, Optional
from llama import Llama, Dialog
import time
from utils.utils import get_bounding_box, process_pcd

def compute_spatial_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    计算检测结果与物体之间的空间相似度
    
    输入:
        DetectionList: 新的检测结果，M个物体
        objects: 地图中的N个物体
    返回:
        MxN 空间相似性
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')
    spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    return spatial_sim


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    计算两组轴线对齐的三维边界框之间的 IoU。
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    返回: MxN 空间相似性
    '''
    # 计算box轴对齐的框架
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)
    # 扩大尺寸
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)
    # 计算最小值的最大值和最大值的最小值，得出交点框的坐标
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)
    # 计算交叉bbox的体积
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)
    # 计算两bbox的体积
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)
    # 计算 IoU，通过将交点体积设置为0来处理没有交点的特殊情况。
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)
    # print(inter_vol)
    # print(iou)
    # print(bbox1)
    # print(bbox2)
    return iou

def compute_visual_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    计算caption相似度
    输入输出与上面的几何相似度一样
    '''
    det_fts = detection_list.get_stacked_values_torch('caption') # (M, D)
    obj_fts = objects.get_stacked_values_torch('caption') # (N, D)
    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    return visual_sim

def preprocess_text(text):
    '''
    caption分词，去除常见冠词等
    '''
    # 移除常见冠词和介词
    stop_words = set(["a","with", "the", "in", "on", "ahead", "is", "to", "next", "down", "of", "along", "and"]) 
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def compute_caption_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    计算caption之间的相似性
    '''
    sentences1 = detection_list.get_stacked_str_torch('caption') # (M, D)
    sentences2 = objects.get_stacked_str_torch('caption') # (N, D)
    # 预处理文本
    processed_sentences1 = [preprocess_text(sentence) for sentence in sentences1]
    processed_sentences2 = [preprocess_text(sentence) for sentence in sentences2]
    # # 使用词袋模型表示文本
    # vectorizer = CountVectorizer().fit_transform(processed_sentences1 + processed_sentences2)
    # 使用TF-IDF向量化文本，这个更好用
    vectorizer = TfidfVectorizer().fit(processed_sentences1 + processed_sentences2)
    vectorizer = vectorizer.transform(processed_sentences1 + processed_sentences2)

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(vectorizer[:len(sentences1)], vectorizer[len(sentences1):])
    # DEBUG查看caption相似性如何
    # for i in range(similarity_matrix.shape[0]):
    #     for j in range(similarity_matrix.shape[1]):
    #         if (similarity_matrix[i,j]>0):
    #             print(sentences1[i])
    #             print(sentences2[j])
    #             print(similarity_matrix[i,j])
    return similarity_matrix


def compute_ft_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    计算caption特征之间的相似性
    '''
    det_fts = detection_list.get_stacked_values_torch('ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('ft') # (N, D)
    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    ft_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    # print(f'ft_sim shape = {ft_sim.shape}')
    return ft_sim

def aggregate_similarities(cfg, spatial_sim: torch.Tensor,  ft_similarities: torch.Tensor, caption_similarities: torch.Tensor) -> torch.Tensor:
    '''
    将空间和caption、caption特征相似性汇总为单一相似性得分
    '''
    # 计算方式可以修改
    if caption_similarities is not None:
        sims = spatial_sim*cfg.spatial_weight + caption_similarities*cfg.caption_weight + ft_similarities*cfg.ft_weight # (M, N)
    else:
        sims = spatial_sim*cfg.spatial_weight + ft_similarities*cfg.ft_weight # (M, N)
    return sims


def merge_detections_to_objects(
    cfg,
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    '''
    把单帧融入地图
    '''
    # 遍历所有检测结果并将其合并为object
    for i in range(agg_sim.shape[0]):
        # 如果未与任何对象匹配，则将其添加为新对象
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # 与最相似的现有对象合并
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det)
            objects[j] = merged_obj
    return objects



def merge_obj2_into_obj1(cfg, obj1, obj2, bg=False, class_name = None):
    '''
    将新对象与旧对象合并，此操作在此完成
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    for k in obj1.keys():
        if k in ['caption']:
            obj1['caption'] += ', '
            obj1['caption'] += obj2['caption']
            if class_name is not None:
                # 背景的cption使用背景的统一caption
                obj1['caption'] = class_name
        elif k not in ['pcd', 'bbox', 'ft', 'bg_class', "class_sk", "captions_ft"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                # 保持原有object的颜色
                obj1[k] = obj1[k] 
            else:
                raise NotImplementedError
        else:
            continue
    # 融合pcd和bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(cfg, obj1['pcd'], use_db=not bg)
    obj1['bbox'] = get_bounding_box(obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    # 融合特征
    obj1['ft'] = (obj1['ft'] * n_obj1_det +
                obj2['ft'] * n_obj2_det) / (
                n_obj1_det + n_obj2_det)
    obj1['ft'] = F.normalize(obj1['ft'], dim=0)
    return obj1


def caption_merge(cfg, objects: MapObjectList):
    '''
    将多个caption融合为一个
    '''
    # LLAMA预训练模型载入
    generator = Llama.build(
        ckpt_dir=cfg.llama_ckpt_dir,
        tokenizer_path=cfg.llama_tokenizer_path,
        max_seq_len=cfg.llama_max_seq_len,
        max_batch_size=cfg.llama_max_batch_size,
    )
    # 用作示范的prompt example
    caption_example1 = "a car parked on the street, a car parked on the street, a white car parked on the street, a car parked on the street, a black car parked on the street, a white car parked on the street, a white car on the road, a mirror of a white car"
    caption_example2 = "a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign"
    caption_example3 = "a triangular street sign, the back of a triangular sign"

    for i in trange(len(objects)):
        caption_obj = objects[i]["caption"]
        if ", " in caption_obj:
            # 如果观测次数太多，字符串会太长，则需要删除一些前面的观测
            comma_count = caption_obj.count(', ')
            if comma_count > cfg.max_caption_num:
                num_last_comma_index = 0
                for j in range(cfg.max_caption_num):
                    num_last_comma_index = caption_obj.rfind(', ', 0, num_last_comma_index - 1)
                # 删除倒数第三个逗号前面的内容
                caption_obj = caption_obj[num_last_comma_index + 2:]
            # 生成llama对话
            dialogs: List[Dialog] = [
                [{"role": "system", 
                "content": "You are a phrase summarizer who can summarize a most complete phrase that best represents \
                them from a sequence of phrases separated by commas, including as much effective information, adjective \
                and elements as possible without severe conflicting. \
                You only need to produce a string of summarized phrase.\
                Please produce nothing else!!!!!!!! Only one phrase. \
                The output format is: 'Summarized parase: [[your summarized parase itself]]'"}
                ,{"role": "user", "content": caption_example1}
                ,{"role": "assistant", "content": "Summarized parase: [a white car parked on the street]"}
                ,{"role": "user", "content": caption_example2}
                ,{"role": "assistant", "content": "Summarized parase: [a red and white sign]"}
                ,{"role": "user", "content": caption_example3}
                ,{"role": "assistant", "content": "Summarized parase: [the back of a triangular street sign]"}
                ,{"role": "user", "content": caption_obj}],
            ]

            # llama进行回答
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len= None,
                temperature=0.6,
                top_p=0.9,
            )
            # 读取llama回答结果中的generation content作为caption融合结果
            for dialog, result in zip(dialogs, results):
                input_text = result["generation"]["content"]
                pattern = r'\[([^]]+)\]'  # 匹配方括号中的内容
                match = re.search(pattern, input_text)
                extracted_content = []
                if match:
                    extracted_content = match.group(1)
                objects[i]["caption"] = extracted_content
    return objects, generator


        
def captions_ft(objects: MapObjectList, bg_objects: MapObjectList, sbert_model):
    '''
    提取融合后的caption特征
    '''
    for i in range(len(objects)):
        caption = objects[i]["caption"]
        caption_fts = sbert_model.encode(caption, convert_to_tensor=True)
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        objects[i]["captions_ft"] = caption_fts
    for i in range(len(bg_objects)):
        caption = bg_objects[i]["caption"]
        caption_fts = sbert_model.encode(caption, convert_to_tensor=True)
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        bg_objects[i]["captions_ft"] = caption_fts
    return objects, bg_objects