# OpenGrpahs
OpenGraphs: Open-Vocabulary Hierarchical 3D Scene Graphs in Large-Scale Outdoor Environments

 ## Abstract
Environment maps endowed with sophisticated semantics are pivotal for facilitating seamless interaction between robots and humans, enabling them to effectively carry out various tasks. Open-vocabulary maps, powered by Visual-Language models (VLMs), possess inherent advantages, including multimodal retrieval and open-set classes. However, existing open-vocabulary maps are constrained to closed indoor scenarios and VLM features, thereby diminishing their usability and inference capabilities. Moreover, the absence of topological relationships further complicates the accurate querying of specific instances. In this work, we propose OpenGraphs, a representation of open-vocabulary hierarchical graph structure designed for large-scale outdoor environments.  OpenGraphs initially extracts instances and their captions from visual images using 2D foundation models, encoding the captions with features to enhance textual reasoning. Subsequently, 3D incremental panoramic mapping with feature embedding is achieved by projecting images onto LiDAR point clouds. Finally, the environment is segmented based on lane graph connectivity to construct a hierarchical scene graph. Validation results from real public dataset SemanticKITTI demonstrate that, even without fine-tuning the models, OpenGraphs exhibit the ability to generalize to novel semantic classes and achieve the highest segmentation and query accuracy.
 
<img src="https://github.com/BIT-DYN/OpenGrpahs/blob/master/fig/first.jpg">

## Install
```bash
git clone https://github.com/BIT-DYN/OpenGraphs
```


## Run

### Generate caption and features
```bash
cd OpenGraphs/
python script/main_gen_cap.py
```
### Generate panoramic maps
```bash
python script/main_gen_pc.py
```
### Map Interaction 
```bash
python script/visualize.py
```
### Generate instance-layer scene graph
```bash
python script/build_scenegraph.py
```

Other codes is coming soon...

