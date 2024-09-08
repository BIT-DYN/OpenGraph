<br>
<p align="center">
<h1 align="center"><strong>OpenGraph: Open-Vocabulary Hierarchical 3D Graph Representation in Large-Scale Outdoor Environments</strong></h1>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2403.09412" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 
</p>


 ## 🏠  Abstract
Environment representations endowed with sophisticated semantics are pivotal for facilitating seamless interaction between robots and humans, enabling them to effectively carry out various tasks. Open-vocabulary maps, powered by Visual-Language models (VLMs), possess inherent advantages, including zero-shot learning and support for open-set classes.
However, existing open-vocabulary maps are primarily designed for small-scale environments, such as desktops or rooms, and are typically geared towards limited-area tasks involving robotic indoor navigation or in-place manipulation. They face challenges in direct generalization to outdoor environments characterized by numerous objects and complex tasks, owing to limitations in both understanding level and map structure.
In this work, we propose OpenGraph, the first open-vocabulary hierarchical graph representation designed for large-scale outdoor environments. 
OpenGraph initially extracts instances and their captions from visual images, enhancing textual reasoning by encoding them. Subsequently, it achieves 3D incremental object-centric mapping with feature embedding by projecting images onto LiDAR point clouds. Finally, the environment is segmented based on lane graph connectivity to construct a hierarchical graph. Validation results from public dataset SemanticKITTI demonstrate that, OpenGraph achieves the highest segmentation and query accuracy.
 
<img src="https://github.com/BIT-DYN/OpenGraph/blob/master/fig/first.jpg">

## 🛠  Install

### Install the required libraries
Use conda to install the required environment. To avoid problems, it is recommended to follow the instructions below to set up the environment.


```bash
conda create -n opengraph anaconda python=3.10
conda activate opengraph

# Install the required libraries
pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy

# Install the Faiss library (CPU version should be fine)
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

##### Install Pytorch according to your own setup #####
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
# conda install pytorch3d -c pytorch3d # This detects a conflict. You can use the command below, maybe with a different version
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
```


###  Install TAG2TEXT Model

```bash
mkdir third_parties & cd third_parties
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

Download pretrained weights
```bash
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```


###  Install Grounding DINO Model

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install --no-build-isolation -e GroundingDINO
```

Download pretrained weights
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```



###  Install TAP Model
Follow the [instructions](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#installation) to install the TAP model and download the pretrained weights [here](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#models).


###  Install SBERT Model
```bash
pip install -U sentence-transformers
```
Download pretrained weights
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

###  Install Llama
Follow the [official installation](https://github.com/meta-llama/llama) of Llama2 to install it.


###  Install 4DMOS
Follow the [official installation](https://github.com/PRBonn/4DMOS.git) of 4DMOS to install it.

We use the [weight 10_scans.ckpt](https://www.ipb.uni-bonn.de/html/projects/4DMOS/10_scans.zip)



### Clone this repo

```bash
git clone https://github.com/BIT-DYN/OpenGraph
cd OpenGraph
```

### Modify the configuration file
You should modify the configuration file ```config/semantickitti.yaml``` according to the address of each file you just installed.

## 📊 Prepare dataset
OpenGraph has completed validation primarily on SemanticKITTI. 

Please download their data from the former [official website](http://www.semantic-kitti.org/
), which can be any sequence of them. 


## 🏃 Run

### Generate caption and features
Run the following command to output the results of instance detection and feature extraction against the image.
```bash
python script/main_gen_cap.py
```
### Generate panoramic maps
Run the following command to complete the incremental build of the 3D map.
```bash
torchrun --nproc_per_node=1 script/main_gen_pc.py
```

### Generate instance-layer scene graph
```bash
python script/build_scenegraph.py
```

### Map Interaction 
Interactions can be made using our visualization files.
```bash
python script/visualize.py
```
Then in the open3d visualizer window, you can use the following key callbacks to change the visualization.

Press ```B``` to toggle the background point clouds (wall, floor, ceiling, etc.).

Press ```C``` to color the point clouds by the object class from the tagging model. 

Press ```R``` to color the point clouds by RGB.

Press ```F``` and type text in the terminal, and the point cloud will be colored by the CLIP similarity with the input text.

Press ```I``` to color the point clouds by object instance ID.

Press ```G``` to visualize the instance-level scene graph.

### Building Hierarchical Graph
Reads historical track points in two dimensions and generates a topology map of the lane graph.
```bash
python script/gen_lane.py
```
Generate a semantic point cloud pcd file for the entire sequence.
```bash
python script/gen_all_pc.py
```
Visualize the hierarchical scene graph of the final build.
```bash
python script/hierarchical_vis.py
```
Then in the open3d visualizer window, you can use the following key callbacks to change the visualization.

Press ```V``` to save current view_params.

Press ```X``` to load saved view_params.

Press ```I``` to vis instance bbox.

Press ```G``` to vis the lines among instance bbox.

Press ```O``` to vis the left layers.

Press ```L``` to vis the lines among layers.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{10638699,
  title={OpenGraph: Open-Vocabulary Hierarchical 3D Graph Representation in Large-Scale Outdoor Environments}, 
  author={Deng, Yinan and Wang, Jiahui and Zhao, Jingyu and Tian, Xinyu and Chen, Guangyan and Yang, Yi and Yue, Yufeng},
  journal={IEEE Robotics and Automation Letters}, 
  year={2024},
  volume={9},
  number={10},
  pages={8402-8409},
}
```

## 👏 Acknowledgements
We would like to express our gratitude to the open-source projects and their contributors [Concept-Graph](https://github.com/concept-graphs/concept-graphs). 
Their valuable work has greatly contributed to the development of our codebase.


