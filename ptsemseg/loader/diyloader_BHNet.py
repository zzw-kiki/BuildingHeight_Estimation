import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import tifffile as tif
from ptsemseg.augmentations.diyaugmentation import my_segmentation_transforms, my_segmentation_transforms_crop
import random
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return np.asarray(Image.open(path))

def stretch_img(image, nrange):
    h, w, nbands = np.shape(image)
    image_stretch = np.zeros(shape=(h, w, nbands), dtype=np.float32)
    for i in range(nbands):
        image_stretch[:, :, i] = 1.0*(image[:, :, i]-nrange[1, i])/(nrange[0, i]-nrange[1, i])
    return image_stretch

def readtif(name):
    img=tif.imread(name)
    return img

class myImageFloder(data.Dataset):
    def __init__(self, imgpath, labpath,augmentations=False, num= 0):
        self.imgpath = imgpath
        self.labpath = labpath
        if num>0:
            self.imgpath = imgpath[:num]
            self.labpath = labpath[:num]
        self.augmentations = augmentations

    def __getitem__(self, index):
        S1path_ = self.imgpath[index, 0]
        S2path_ = self.imgpath[index, 1]
        POIpath_ = self.imgpath[index, 2]
        labpath_ = self.labpath[index]
        # print(muxpath_)
        S1 = tif.imread(S1path_)  # already convert to 0-1 from gee
        S2 = tif.imread(S2path_) # already convert to 0-1 from gee
        POI = tif.imread(POIpath_)*1000 # convert to 0-1
        img = np.concatenate((S1,S2,POI), axis=2).astype(np.float32)  # the third dimension
        img[img>1]=1 # ensure data range is 0-1
        lab = tif.imread(labpath_).astype(np.float32)

        if self.augmentations:
            img, lab = my_segmentation_transforms(img, lab)
        img = img.transpose((2, 0, 1)) # H W C => C H W
        lab = np.expand_dims(lab, axis=0)
        img = torch.tensor(img.astype(np.float32), dtype=torch.float32)
        lab = torch.tensor(lab.astype(np.float32), dtype=torch.float32)
        return img, lab

    def __len__(self):
        return len(self.imgpath)

class myImageFloder_HR(data.Dataset):
        def __init__(self, imgpath, labpath, augmentations=False, num=0):
            self.imgpath = imgpath
            self.labpath = labpath
            if num > 0:
                self.imgpath = imgpath[:num]
                self.labpath = labpath[:num]
            self.augmentations = augmentations

        def __getitem__(self, index):
            S1path_ = self.imgpath[index, 0]
            S2path_ = self.imgpath[index, 1]
            POIpath_ = self.imgpath[index, 2]
            RGBNpath_ = self.imgpath[index, 3]
            multiviewpath_ = self.imgpath[index, 4]
            labpath_ = self.labpath[index]
            # print(muxpath_)
            S1 = tif.imread(S1path_)  # already convert to 0-1 from gee
            S2 = tif.imread(S2path_)  # already convert to 0-1 from gee
            POI = tif.imread(POIpath_) * 1000  # convert to 0-1
            multi_spectral = tif.imread(RGBNpath_) * 1000
            multi_view = tif.imread(multiviewpath_) * 1000
            img = np.concatenate((S1, S2, POI,multi_spectral,multi_view), axis=2).astype(np.float32)  # the third dimension
            img[img > 1] = 1  # ensure data range is 0-1
            lab = tif.imread(labpath_).astype(np.float32)

            if self.augmentations:
                img, lab = my_segmentation_transforms(img, lab)
            img = img.transpose((2, 0, 1))  # H W C => C H W
            lab = np.expand_dims(lab, axis=0)
            img = torch.tensor(img.astype(np.float32), dtype=torch.float32)
            lab = torch.tensor(lab.astype(np.float32), dtype=torch.float32)
            return img, lab

        def __len__(self):
            return len(self.imgpath)

