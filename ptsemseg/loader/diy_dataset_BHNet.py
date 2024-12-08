import os
import numpy as np
from os.path import join

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.tif'  #new added
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    image = list()
    labels = list()

    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        for root, dirs, files in os.walk(filepath):
            if root.endswith('img'):
                for name in files:
                    ipath = os.path.join(root, name)
                    image.append(ipath)
            elif root.endswith('lab'):
                for name in files:
                    ipath = os.path.join(root, name)
                    labels.append(ipath)
    assert len(image) == len(labels)
    image.sort()
    labels.sort()
    image = np.array(image)
    labels = np.array(labels)
    return image, labels


def dataloaderbh(filepath, split=[0.7, 0.1,0.2]):
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        S1 = [join(filepath, 'S1', name) for name in os.listdir(join(filepath, 'S1'))]
        lab = [join(filepath, 'lab_16%', name) for name in os.listdir(join(filepath, 'lab_16%'))]
        S2 = [join(filepath, 'S2', name) for name in os.listdir(join(filepath, 'S2'))]
        POI = [join(filepath, 'POI', name) for name in os.listdir(join(filepath, 'POI'))]

    assert len(S1) == len(lab)
    assert len(S1) == len(S2)
    assert len(S1) == len(POI)
    S1.sort()
    S2.sort()
    POI.sort()
    lab.sort()

    num_samples=len(S1)
    lab=np.array(lab)
    S1 =np.array(S1)
    S2 = np.array(S2)
    POI = np.array(POI)
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0]) # the same as floor
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train+num_val)]
    # test = seq[num_train:]

    imgt = np.vstack((S1[train], S2[train], POI[train])).T
    labt = lab[train]

    imgv = np.vstack((S1[val], S2[val], POI[val])).T
    labv = lab[val]

    return imgt, labt, imgv, labv

def dataloaderbh_HR(filepath, split=[0.7, 0.1,0.2]):
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        S1 = [join(filepath, 'S1', name) for name in os.listdir(join(filepath, 'S1'))]
        lab = [join(filepath, 'lab_16%', name) for name in os.listdir(join(filepath, 'lab_16%'))]
        S2 = [join(filepath, 'S2', name) for name in os.listdir(join(filepath, 'S2'))]
        POI = [join(filepath, 'POI', name) for name in os.listdir(join(filepath, 'POI'))]
        Multi_spectral = [join(filepath, 'multi-spectral', name) for name in os.listdir(join(filepath, 'multi-spectral'))]
        Multi_view = [join(filepath, 'multi-view', name) for name in os.listdir(join(filepath, 'multi-view'))]

    assert len(S1) == len(lab)
    assert len(S1) == len(S2)
    assert len(S1) == len(POI)
    assert len(S1) == len(Multi_spectral)
    assert len(S1) == len(Multi_view)
    S1.sort()
    S2.sort()
    POI.sort()
    lab.sort()
    Multi_spectral.sort()
    Multi_view.sort()

    num_samples=len(S1)
    lab=np.array(lab)
    S1 =np.array(S1)
    S2 = np.array(S2)
    POI = np.array(POI)
    Multi_spectral = np.array(Multi_spectral)
    Multi_view = np.array(Multi_view)

    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0]) # the same as floor
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train+num_val)]

    imgt = np.vstack((S1[train], S2[train], POI[train], Multi_spectral[train], Multi_view[train])).T
    labt = lab[train]

    imgv = np.vstack((S1[val], S2[val], POI[val], Multi_spectral[train], Multi_view[train])).T
    labv = lab[val]

    return imgt, labt, imgv, labv

