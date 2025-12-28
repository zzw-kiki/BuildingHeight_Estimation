An attention-based deep learning framework for building height estimation fusing multimoda data
===
BHNet——an attention-based deep learning framework for building height estimation fusing multimoda data, consists of multiple parallel U-Net structures, with each U-Net dedicated to feature extraction for a specific data source. The model utilizes Sentinel-1 (S1) radar data, Sentinel-2 (S2) multispectral imagery, and POI data, processing each data type separately before fusing the results to generate the final height map.Each input channel is processed through a dedicated encoder-decoder pathway, incorporating channel attention, spatial attention mechanisms and residual blocks to enhance feature learning. The outputs of these pathways are fused to produce the final 10m resolution building height prediction map.
![ilovepdf_merged_page-0003](https://github.com/user-attachments/assets/cf2f8c5f-6b1f-47ed-8de8-ebc7dd64df36)
![image](https://github.com/user-attachments/assets/5f555d55-c7c2-437b-8708-ffb98ed5d8f6)
![image](https://github.com/user-attachments/assets/5c094508-ee6e-4dfd-90a5-2303e10d1254)

1.Dataset & Gee processing
----
* We first used the random forest method on Google Earth Engine to perform variable selection, preprocessing, and data download. Based on the feature importance ranking from the random forest, we selected 20 feature variables derived from Sentinel-1 (Collection Snippet: "COPERNICUS/S1_GRD") and Sentinel-2 (Collection Snippet: "COPERNICUS/S2_SR_HARMONIZED") datasets available on the GEE platform. The preprocessing of feature variables included annual averaging, annual maximum calculations, multi-window statistics, normalization, resampling, and cloud removal. The spatial extent was defined by a locally uploaded vector boundary of Shanghai, and the temporal range was set to 2019 and 2023.
For socioeconomic data, we collected POI data for Shanghai from 2019 and 2023, categorized into four main types: companies and enterprises, catering, shopping services, and accommodation
services. We then performed kernel density analysis to generate raster images with a resolution of 10 meters, which was completed using ArcGIS Pro.
![image](https://github.com/user-attachments/assets/1f78d745-2e2d-48e1-81b8-234178601cbb)
* For reference building heights, we collected building vector footprint data from Baidu Maps for 2019 and 2023 and converted it into raster data with a 10m resolution. All input data was spatially aligned, unified into the same spatial coordinate system, and high-quality samples were selected as training data. These samples covered diverse scenarios, various building morphologies, both clustered and isolated buildings. A 4km×4km grid of Shanghai, corresponding to 400×400 pixels, was created. The selected high-quality samples were then sliced and exported as TIFF files to serve as the training dataset.We use data from 2023 for training and validate the model's performance on data from 2019.

The sample data is stored in the **sample-BHNet** directory. The **S1**, **S2**, and **POI** folders contain preprocessed and downloaded data from GEE, including 2-band Sentinel-1 data, 14-band Sentinel-2 data, and 4-band POI composite data, respectively. The **lab** folder stores reference building height data obtained from Baidu Maps, representing the reference building heights with a 2% interval after removing the maximum height.Among these, we only uploaded the sample lab data after removing the top **10%** of the maximum heights.The entire process of creating the training dataset was completed in ArcGIS Pro.

2.Training and predcition
----
* The training code is **train_BHNet.py**, and the weight files are stored in **runs/BHNet_10m_10%_300epoch/finetune_298.rar**. We only uploaded the weights trained with the reference building heights after removing the top 10% of the maximum heights. The **log** folder contains the training logs of 300 epoches using different reference building heights, including the overall training loss, as well as the training losses and RMSE for each.


![image](https://github.com/user-attachments/assets/1e1b07a9-9825-415f-89e7-80c52c1b047a)![image](https://github.com/user-attachments/assets/89a27346-0142-4ae6-9327-def4977f2d9b)
* Perform global prediction: **BHNet_predict.ipynb**, slice the input data of Shanghai into 400×400 pixel segments and then stitch the results together.

3.Evaluation
---
For the entire Shanghai area, 100,000 random points are selected for accuracy validation. The corresponding values from the reference building height layer and the predicted image are extracted for each point, and the results are output to a CSV file. For the district and street-level spatial scales within Shanghai, the same method is applied, but all the pixel points within the region are used for validation. The accuracy validation is carried out in **evaluation.py** and **evaluation_in_different_scales.py**.

4.BHNet-HR
---
* To improve the prediction of building height details and explore the network's performance on high-resolution remote sensing imagery, we optimized the original network and designed the BHNet-HR model.
![image](https://github.com/user-attachments/assets/d3b95e7b-1c05-4181-9e74-0a47e07e7996)
*  The 2019 data comes from the ZY3-01 satellite, with a 2.1m resolution for nadir panchromatic, 3.5m for forward and backward panchromatic, and 5.8m for the four-band multispectral (RGBN). The 2023 data comes from the ZY3-02 satellite, with 2.1m nadir panchromatic, 2.5m forward and backward panchromatic, and 5.8m multispectral resolution. Data preprocessing includes radiometric correction, orthorectification, image registration, and fusion. All images were resampled to 2.5m. The multispectral images were fused with panchromatic images using the Gram-Schmidt method to enhance spatial detail. To reduce radiometric differences between multi-view images, a histogram matching algorithm was used to normalize forward and backward images to the nadir view. The input data and labels were resampled to a 2.5m resolution, and the network was modified by adding two channels for multispectral (RGBN) and multi-view (nads, fwds, bwds) data. The model was trained on the 2023 dataset to generate high-resolution prediction images.The sample data is stored in the **sample-BHNet-HR** directory.

5.Result
---
* The prediction results for 2019 and 2023 are stored in the **Result** folder, with the projection coordinate system set to WGS84 UTM51N. For pixels with predicted values less than 0, we set them to 0.
![image](https://github.com/user-attachments/assets/f946d02b-c1c9-4272-869c-d4cf558ee2ec)
![image](https://github.com/user-attachments/assets/73848ff6-74a5-4148-87f1-2cf102f1cff4)
![image](https://github.com/user-attachments/assets/318f6171-d780-4bdf-9755-e75e44265c34)
![image](https://github.com/user-attachments/assets/917caba5-3b5f-45d8-81f9-bdcfdc4200ea)
![image](https://github.com/user-attachments/assets/e82f913a-0b4a-4308-ba4e-30674c75e742)



