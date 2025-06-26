# Bidirectional skip-frame prediction for video anomaly detection with intra-domain disparity-driven attention (BiSP)

Published in [Pattern Recognition 2025] 
python 3.8  
torch 1.13  

0. Data preparation      
Ped1/Ped2: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm  
Avenue: https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html  
ShanghaiTech: https://svip-lab.github.io/dataset/campus_dataset.html  

1. Train  
python Train.py # Ped2  
python Train_ped1.py # Ped1  
python Train_avenue.py # Avenue  
python Train_shanghaitech.py # ShanghaiTech  
  
3. Evaluation  
python Evaluate.py # Ped2  
python Evaluate_ped1.py # Ped1  
python Evaluate_avenue.py # Avenue  
python Evaluate_shanghaitech.py # ShanghaiTech  

You can download the pretrained weights of BiSP for the four datasets from [Google](https://drive.google.com/drive/folders/1Vcs2mryGiZmidjaQy1C0Elviv1ADzBru?usp=sharing).

If you use this work, please cite:
```
@article{Lyu2025Bisp,
title = {Bidirectional skip-frame prediction for video anomaly detection with intra-domain disparity-driven attention},
author = {Jiahao Lyu and Minghua Zhao and Jing Hu and Runtao Xi and Xuewen Huang and Shuangli Du and Cheng Shi and Tian Ma},
journal = {Pattern Recognition},
pages = {112010},
year = {2025},
issn = {0031-3203}
}
```
