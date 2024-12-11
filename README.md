BHNet: A Multi-Task Convolutional Neural Network Specifically Designed for Predicting Building Heights from Multispectral and Socioeconomic Data
===
BHNet consists of multiple parallel U-Net structures, with each U-Net dedicated to feature extraction for a specific data source. The model utilizes Sentinel-1 (S1) radar data, Sentinel-2 (S2) multispectral imagery, and POI data, processing each data type separately before fusing the results to generate the final height map.Each input channel is processed through a dedicated encoder-decoder pathway, incorporating channel attention, spatial attention mechanisms and residual blocks to enhance feature learning. The outputs of these pathways are fused to produce the final 10m resolution building height prediction map.
![ilovepdf_merged_page-0003](https://github.com/user-attachments/assets/cf2f8c5f-6b1f-47ed-8de8-ebc7dd64df36)
![ilovepdf_merged_page-0006](https://github.com/user-attachments/assets/d8e2679f-a1a5-4cab-833b-9427e5f1fe94)

1.Dataset & Gee processing
----
We first used the random forest method on Google Earth Engine to perform variable selection, preprocessing, and data download. Based on the feature importance ranking from the random forest, we selected 20 feature variables derived from Sentinel-1 (Collection Snippet: "COPERNICUS/S1_GRD") and Sentinel-2 (Collection Snippet: "COPERNICUS/S2_SR_HARMONIZED") datasets available on the GEE platform. The preprocessing of feature variables included annual averaging, annual maximum calculations, multi-window statistics, normalization, resampling, and cloud removal. The spatial extent was defined by a locally uploaded vector boundary of Shanghai, and the temporal range was set to 2019 and 2023.
For socioeconomic data, we collected POI data for Shanghai from 2019 and 2023, categorized into four main types: companies and enterprises, catering, shopping services, and accommodation
services. We then performed kernel density analysis to generate raster images with a resolution of 10 meters, which was completed using ArcGIS Pro.
![image](https://github.com/user-attachments/assets/1f78d745-2e2d-48e1-81b8-234178601cbb)
For reference building heights, we collected building vector footprint data from Baidu Maps for 2019 and 2023 and converted it into raster data with a 10m resolution. All input data was spatially aligned, unified into the same spatial coordinate system, and high-quality samples were selected as training data. These samples covered diverse scenarios, various building morphologies, both clustered and isolated buildings. A 4km×4km grid of Shanghai, corresponding to 400×400 pixels, was created. The selected high-quality samples were then sliced and exported as TIFF files to serve as the training dataset.

The sample data is stored in the "sample-BHNet" directory. The S1, S2, and POI folders contain preprocessed and downloaded data from GEE, including 2-band Sentinel-1 data, 14-band Sentinel-2 data, and 4-band POI composite data, respectively. The lab folder stores reference building height data obtained from Baidu Maps, representing the reference building heights with a 2% interval after removing the maximum height. The entire process of creating the training dataset was completed in ArcGIS Pro.







