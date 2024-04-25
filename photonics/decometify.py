import numpy as np

from .utils import PROJECT_ROOT

directory_240424 = PROJECT_ROOT + "/data/pl_240424"



random_images = np.load(directory_240424 + "/pl_240425_002600_669261.npy")
xc = [469, 541, 496, 428, 396, 423, 609, 599, 555, 486, 407, 354, 323, 334, 372, 442, 512, 572]
yc = [308, 305, 375, 366, 310, 243, 286, 365, 420, 451, 441, 394, 322, 249, 198, 166, 166, 214]
s = 25

def softmax(arr):
    arr = arr - np.max(arr)
    arr = np.exp(arr)
    arr = arr / np.sum(arr)
    return arr

"""circle_cutout = np.zeros((2*s,2*s))
yi, xi = np.indices(circle_cutout.shape)
circle_cutout[np.where((yi - s) ** 2 + (xi - s) ** 2 <= s ** 2)] = 1
masks = [circle_cutout for _ in range(len(xc))]"""
masks = [softmax(np.std(random_images[:,ycv-s:ycv+s,xcv-s:xcv+s], axis=0)) for (xcv, ycv) in zip(xc, yc)]

def intensities_from_comet(img):
    intensities = np.zeros(18)
    for (i, (xcv, ycv, m)) in enumerate(zip(xc, yc, masks)):
        cutout = img[ycv-s:ycv+s,xcv-s:xcv+s]
        intensities[i] = np.sum(cutout * m)
        
    return intensities
