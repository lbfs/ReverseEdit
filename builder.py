import time
import pickle
import cv2
from tqdm import tqdm
from vptree import VPTree
from clip import ClipReader
from image import perceptual_hash, hamming_distance, compare_ssim, crop_image_only_outside

class ProcessedFrameInfo:
    def __init__(self):
        self.position = None
        self.filename = None
        self.hash = None
        self.nearest_neighbors = None
        self.best_neighbor = None

def process_frames(clip):
    processed_frames = []
    for position, frame in enumerate(tqdm(clip, total=len(clip))):
        processed = ProcessedFrameInfo()
        adjusted_frame = crop_image_only_outside(frame, tol=30) # TODO: adjust?
        if adjusted_frame.size < 32 * 32 + 1:
            adjusted_frame = frame
        processed.hash = perceptual_hash(frame)
        processed.position = position
        processed.filename = clip.filename
        processed_frames.append(processed)
    return processed_frames

def find_nearest_matches(tree, processed_edit_frames, depth=5):
    for processed in tqdm(processed_edit_frames):
        nearest_neighbors = tree.get_n_nearest_neighbors(processed, depth)
        nearest_neighbors = [element[1] for element in nearest_neighbors]
        processed.nearest_neighbors = nearest_neighbors
        processed.best_neighbor = nearest_neighbors[0]

def find_closest_ssims_in_nearest_neighbors(edit_clip, source_clips, processed_edit_frames):
    for position, processed in enumerate(tqdm(processed_edit_frames)):
        if len(processed.nearest_neighbors) == 1:
            processed.best_neighbor = processed.nearest_neighbors[0]
            continue
        
        edit_frame = edit_clip[position]
        shape = edit_frame.shape
        new_shape = (shape[0] // 2, shape[1] // 2) #Constant TODO: Look at SSIM Spec for best thing to do here
        edit_frame = cv2.resize(edit_frame, new_shape)

        best_neighbor, best_result = None, 0
        for neighbor in processed.nearest_neighbors:
            source_clip = source_clips[neighbor.filename]
            resized_neighbor_frame = cv2.resize(source_clip[neighbor.position], new_shape)
            res = compare_ssim(edit_frame, resized_neighbor_frame, multichannel=True)
            if res > best_result:
                best_neighbor = neighbor
                best_result = res
        processed.best_neighbor = best_neighbor

def hamming_distance_processed(u, v):
    return hamming_distance(u.hash, v.hash)

def debug_export(clips, processed_frames):
    pass

if __name__ == "__main__":
    edit_filename = "../time.mkv"
    source_filenames = ["../halo720.mp4"]

    start_time = time.time()
    print("Started the process at", start_time)

    edit_clip = ClipReader(edit_filename)
    source_clips = {}
    for filename in source_filenames:
        source_clips[filename] = ClipReader(filename)

    print("Processing edit clip.")
    processed_edit_frames = process_frames(edit_clip)

    print("Processing source clips.")
    processed_source_frames = []
    for filename in source_clips:
        processed_source_frames.extend(process_frames(source_clips[filename]))

    print("Build Tree")
    tree = VPTree(processed_source_frames, hamming_distance_processed)

    print("Find best matches")
    find_nearest_matches(tree, processed_edit_frames)
    #find_closest_ssims_in_nearest_neighbors(edit_clip, source_clips, processed_edit_frames)

    end_time = time.time()
    print("Ended the process at", end_time, "and it took", end_time - start_time)

    with open("export.pickle", "wb") as f:
        pickle.dump(processed_edit_frames, f)

    print("Displaying preview!")

    first = True
    for processed in processed_edit_frames:
        neighbor = processed.best_neighbor
        mat = source_clips[neighbor.filename][neighbor.position]
        cv2.imshow("Video2", mat)
        if first:
            cv2.waitKey(0)
            first = False
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    print("Exporting data")

