import vptree
import cv2
from clip import ClipReader
from image import perceptual_hash, hamming_distance

if __name__ == "__main__":
    clip = ClipReader("../hawkling.mkv")

    hashes = []
    for frame in clip:
        h = perceptual_hash(frame)
        hashes.append(h)

    tree = vptree.VPTree(hashes, hamming_distance)
    print(tree.get_nearest_neighbor(perceptual_hash(clip[788])))

    cv2.destroyAllWindows()