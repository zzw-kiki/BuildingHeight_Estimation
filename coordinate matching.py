import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# 输入文件路径
no_reference_tif_path = 'E:/part_regions/2019_10_2×2/pred/22.tif'
reference_tif_path = 'E:/part_regions/2019_10_2×2/22_S2.tif'
output_tif_path = 'E:/part_regions/2019_10_2×2/pred/22_geo.tif'

# 读取具有坐标参考的 TIFF 文件的元数据
with rio.open(reference_tif_path) as src:
    reference_meta = src.meta.copy()

# 读取没有坐标参考的 TIFF 文件的数据
with rio.open(no_reference_tif_path) as src:
    no_reference_data = src.read()
    no_reference_meta = src.meta.copy()

# 更新没有坐标参考的 TIFF 文件的元数据
# 将其更新为具有坐标参考的 TIFF 文件的元数据
no_reference_meta.update({
    'crs': reference_meta['crs'],
    'transform': reference_meta['transform'],
    'width': reference_meta['width'],
    'height': reference_meta['height']
})

# 写入更新后的 TIFF 文件
with rio.open(output_tif_path, 'w', **no_reference_meta) as dst:
    dst.write(no_reference_data)

print(f"Output TIFF file with reference saved to: {output_tif_path}")
