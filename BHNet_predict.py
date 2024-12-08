import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import torch
from tqdm import tqdm
import numpy as np
import tifffile as tif
from ptsemseg.metrics import heightacc
from ptsemseg.models import BHNet
import matplotlib.pyplot as plt
import tifffile as tif
from os.path import join
import math
import time
from osgeo import gdal
import rasterio as rio

# Setup device
device = 'cuda'
# Setup Model
model = BHNet(n_classes=1).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
resume = r'runs\BHNet_10m_10%_300epoch\finetune_298.tar'
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at resume")
    print("=> Will start from scratch.")
model.eval()
def predict_whole_image(model, image, r, c, grid=400):
    # grid=400
    n, b, rows, cols = image.shape
    # rows=math.ceil(r/grid)*grid
    # cols=math.ceil(c/grid)*grid
    # image_= np.pad(image,((0,0), (0,0), (0,rows-r), (0,cols-c)),'symmetric')
    # weight = np.ones((rows, cols))
    res = np.zeros((rows, cols), dtype=np.float32)
    num_patch = len(range(0, rows, grid)) * len(range(0, cols, grid))
    print('num of patch is', num_patch)
    k = 0
    for i in range(0, rows, grid):
        for j in range(0, cols, grid):
            patch = image[0:, 0:, i:i + grid, j:j + grid].astype('float32')
            if np.max(patch.flatten()) <= 10e-8:
                continue
            start = time.time()
            patch = torch.from_numpy(patch).float()
            pred = model(patch.to(device))
            pred0 = pred[0].cpu().detach().numpy()  # height
            res[i:i + grid, j:j + grid] = np.squeeze(pred0)
            end = time.time()
            k = k + 1
            # print('patch [%d/%d] time elapse:%.3f'%(k,num_patch,(end-start)))
    res = res[0:r, 0:c].astype(np.float32)
    return res

def read_filepath(filepath, filename):
    filelist = list()
    for root, dirs, files in os.walk(filepath):
        for name in files:
            if name.endswith(filename):
                filelist.append(join(root, name))
    filelist.sort()
    return filelist

filelistnew = [r'E:\part_regions\2019_10_2×2\22_S1.tif']

grid = 400
def loadenvi(file, band=1):
    dataset = gdal.Open(file)
    band = dataset.GetRasterBand(band)
    data = band.ReadAsArray()
    dataset = None
    band = None
    return data

data = loadenvi(filelistnew[0], band=1)
print(data.shape)
print(data.dtype)
data = None
for file in filelistnew[:1]:
    # filepath
    idirname = os.path.dirname(file)
    predpath = join(idirname, 'pred')
    respath = join(predpath, '22.tif')
    if os.path.exists(respath):
        print('file: %s already exist, then skip' % respath)
        continue
    if not os.path.exists(predpath):
        print('mkdir %s' % predpath)
        os.mkdir(predpath)
    print('process: %s', file)
    S1_path = join(idirname, '22_S1.tif')
    S2_path = join(idirname, '22_S2.tif')
    POI_path = join(idirname, '22_POI.tif')
    # 1.read image
    band1 = loadenvi(file, band=1)
    r, c = band1.shape[:2]
    img = np.zeros((r, c, 20), dtype='float32')
    for i in range(2):
        img[:, :, i] = loadenvi(file, i + 1)
    for i in range(14):
        img[:, :, 2 + i] = loadenvi(S2_path, i + 1)
    for i in range(4):
        img[:, :, 16 + i] = loadenvi(POI_path, i + 1) * 1000.0
    # for i in range(4):
    img = img.transpose(2, 0, 1)  # C H W
    img = np.expand_dims(img, axis=0)  # 1 C H W
    rows = math.ceil(r / grid) * grid
    cols = math.ceil(c / grid) * grid
    img = np.pad(img, ((0, 0), (0, 0), (0, rows - r), (0, cols - c)), 'symmetric')
    # 2. predict
    starttime = time.time()
    res = predict_whole_image(model, img, r, c, grid=400)
    endtime = time.time()
    print('success: %s, time is %f' % (file, endtime - starttime))
    img = 0  # release
    # 3. export
    reffile = file
    rastermeta = rio.open(reffile).profile
    rastermeta.update(dtype=res.dtype, count=1, compress='lzw')
    with rio.open(respath, mode="w", **rastermeta) as dst:
        dst.write(res, 1)
    tif.imwrite(respath, res)  # building height
