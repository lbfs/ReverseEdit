import scipy.fftpack
import scipy.spatial
import numpy as np
import cv2

from skimage.measure import compare_ssim as compare_ssim_library

# Image hashing
def perceptual_hash(frame, hash_size=8, dct_size=32):
    """
    Perceptual algorithm implemented based on the following article
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    dct_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (dct_size, dct_size))
    dct = scipy.fftpack.dct(scipy.fftpack.dct(dct_frame, axis=0), axis=1)
    low = dct[:hash_size, :hash_size]
    med = np.median(low)
    return low > med

def crop_image_only_outside(img,tol=0):
    """
	Cropping algorithm from user Divakar
	https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
	"""
    mask = img>tol
    if img.ndim == 3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()-1
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()-1
    return img[row_start:row_end,col_start:col_end]

# Compute the hamming distance between different image hashes
def hamming_distance(u, v):
    return scipy.spatial.distance.hamming(u.flatten(), v.flatten())

# Structural similiarity index between two frames
def compare_ssim(u, v, multichannel=True):
    return compare_ssim_library(u, v, multichannel=multichannel)
