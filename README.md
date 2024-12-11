BHNet: A Multi-Task Convolutional Neural Network Specifically Designed for Predicting Building Heights from Multispectral and Socioeconomic Data
===
BHNet consists of multiple parallel U-Net structures, with each U-Net dedicated to feature extraction for a specific data source. The model utilizes Sentinel-1 (S1) radar data, Sentinel-2 (S2) multispectral imagery, and POI data, processing each data type separately before fusing the results to generate the final height map.Each input channel is processed through a dedicated encoder-decoder pathway, incorporating channel attention, spatial attention mechanisms and residual blocks to enhance feature learning. The outputs of these pathways are fused to produce the final 10m resolution building height prediction map.
![ilovepdf_merged_page-0003](https://github.com/user-attachments/assets/6bcfaff0-d1b9-4445-86e2-16c8f549c7c1)

1.Dataset & Gee processing
----









