import scipy.fftpack
import scipy.spatial
import numpy as np
import cv2

from skimage.measure import compare_ssim as compare_ssim_library


class HashedFrame:
    __slots__ = ["position", "filename", "hash",
                 "nearest_neighbors", "best_neighbor"]

    def __init__(self):
        self.position = None
        self.filename = None
        self.hash = None
        self.nearest_neighbors = None
        self.best_neighbor = None

    def __hash__(self):
        return hash(("filename", self.filename, "position", self.position))

    def __eq__(self, other):
        return self.position == other.position and self.filename == other.filename

    def __lt__(self, other):
        return self.position < other.position

    @staticmethod
    def compute_distance(u, v):
        return scipy.spatial.distance.hamming(u.hash.flatten(), v.hash.flatten())


class ImageTool:
    @staticmethod
    def perceptual_hash(frame, hash_size=8, dct_size=32):
        """
        Perceptual algorithm implemented based on the following article
        http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        """
        dct_frame = cv2.resize(cv2.cvtColor(
            frame, cv2.COLOR_RGB2GRAY), (dct_size, dct_size))
        dct = scipy.fftpack.dct(scipy.fftpack.dct(dct_frame, axis=0), axis=1)
        low = dct[:hash_size, :hash_size]
        med = np.median(low)
        return low > med

    @staticmethod
    def crop_image_only_outside(img, tolerance=0, minimum_width=None, minimum_height=None):
        """
        Cropping algorithm from user Divakar
        https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
        """
        mask = img > tolerance
        if img.ndim == 3:
            mask = mask.all(2)
        m, n = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()-1
        row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()-1

        # Only use the crop if its above a certain threshold
        img_cropped = img[row_start:row_end, col_start:col_end]
        if minimum_width is not None and minimum_height is not None:
            if row_end - row_start < minimum_height or col_end - col_start < minimum_width:
                # TODO: DEBUG
                return img

        return img_cropped

    @staticmethod
    def compare_ssim(u, v, multichannel=True):
        return compare_ssim_library(u, v, multichannel=multichannel)
