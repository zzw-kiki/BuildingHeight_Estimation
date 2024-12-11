BHNet: A Multi-Task Convolutional Neural Network Specifically Designed for Predicting Building Heights from Multispectral and Socioeconomic Data
===
<div style="text-align: justify;">
BHNet consists of multiple parallel U-Net structures, with each U-Net dedicated to feature extraction for a specific data source. The model utilizes Sentinel-1 (S1) radar data, Sentinel-2 (S2) multispectral imagery, and POI data, processing each data type separately before fusing the results to generate the final height map.Each input channel is processed through a dedicated encoder-decoder pathway, incorporating channel attention, spatial attention mechanisms and residual blocks to enhance feature learning. The outputs of these pathways are fused to produce the final 10m resolution building height prediction map.
</div>
![ilovepdf_merged_page-0003](https://github.com/user-attachments/assets/cf2f8c5f-6b1f-47ed-8de8-ebc7dd64df36)
![ilovepdf_merged_page-0006](https://github.com/user-attachments/assets/d8e2679f-a1a5-4cab-833b-9427e5f1fe94)

1.Dataset & Gee processing
----









