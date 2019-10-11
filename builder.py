import time
import cv2
from tqdm import tqdm
from vptree import VPTree
from clip import ClipReader
from image import perceptual_hash, hamming_distance, HashedFrame

def build_hashes(clip):
    hashes = []
    for index, frame in enumerate(clip):
        hashed_frame = HashedFrame()
        hashed_frame.hash = perceptual_hash(frame)
        hashed_frame.index = index
        hashed_frame.clip = clip
        hashes.append(hashed_frame)
    return hashes

def find_best_matches(tree, hashes):
    best_matches = []
    for hash_value in tqdm(hashes):
        best_match = tree.get_nearest_neighbor(hash_value)
        best_matches.append(best_match[1])
    return best_matches

def find_best_matches_across_videos(edited_video_filename, source_filenames):
    # source filenames is a list of strings
    pass

def do_ssim_accuracy_correction():
    pass

if __name__ == "__main__":
    start_time = time.time()
    print("Started the process at", start_time)

    clip = ClipReader("../hawkling.mkv")
    v_clip = ClipReader("../ark.mkv")

    print("Building hashes for source clip.")
    v_hashes = build_hashes(v_clip)
    print("Building tree for source clip out of source hashes.")
    tree = VPTree(v_hashes, hamming_distance)
    print("Building hashes for edit clip.")
    hashes = build_hashes(clip)
    print("Performing lookup!")
    bests = find_best_matches(tree, hashes)

    end_time = time.time()
    print("Ended the process at", end_time, "and it took", end_time - start_time)

    print("Displaying preview!")

    for hashed_frame in bests:
        mat = hashed_frame.frame
        cv2.imshow("Video2", mat)
        cv2.waitKey(1)

    cv2.destroyAllWindows()