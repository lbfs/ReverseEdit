import scipy.fftpack
import scipy.spatial
import numpy as np
import cv2

from skimage.measure import compare_ssim

# Image hashing
def perceptual_hash(frame, hash_size=8, dct_size=32):
	"""
	Perceptual algorithm implemented based on the following article
	http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
	"""
	dct_frame = cv2.cvtColor(cv2.resize(frame, (dct_size, dct_size)), cv2.COLOR_RGB2GRAY)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(dct_frame, axis=0), axis=1)
	low = dct[:hash_size, :hash_size]
	med = np.median(low)
	return low > med

def average_hash(frame, hash_size=8):
	"""
	Average hash algorithm implemented based on the following article
	http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
	"""
	frame = cv2.cvtColor(cv2.resize(frame, (hash_size, hash_size)), cv2.COLOR_RGB2GRAY)
	return frame > np.mean(frame)

# Structural similiarity index between two frames
def structural_similiarity(u, v, multichannel=True):
    return compare_ssim(u, v, multichannel=multichannel)

# Compute the hamming distance between different image hashes
def hamming_distance(u, v):
    return scipy.spatial.distance.hamming(u.flatten(), v.flatten())
