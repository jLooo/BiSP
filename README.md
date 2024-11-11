# Bidirectional skip-frame prediction for video anomaly detection with intra-domain disparity-driven attention (BiSP)

code for the journal PR  
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
