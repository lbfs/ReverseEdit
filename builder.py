import time
import pickle
import multiprocessing
import cv2
from tqdm import tqdm
from vptree import VPTree
from clip import ClipReader
from image import perceptual_hash, hamming_distance, compare_ssim, crop_image_only_outside

mp_data = None #Used for values that will be reused.

class ProcessedFrameInfo:
    def __init__(self):
        self.position = None
        self.filename = None
        self.hash = None
        self.nearest_neighbors = None
        self.best_neighbor = None
    
    def __hash__(self):
        return hash(('filename', self.filename, 'position', self.position))

    def __eq__(self, other):
        return self.position == other.position and self.filename == other.filename

# Crop and hash frames
def process_frames_init():
    global mp_data
    hash_size = 8
    dct_size = 8 * 4
    window_size = dct_size * dct_size + 1
    mp_data = (hash_size, dct_size, window_size)

def process_frames_actor(data):
    global mp_data
    
    position, frame = data
    processed = ProcessedFrameInfo()

    hash_size, dct_size, window_size = mp_data
    adjusted_frame = crop_image_only_outside(frame, tol=40)

    if adjusted_frame.size < window_size:
        adjusted_frame = frame

    processed.hash = perceptual_hash(adjusted_frame, hash_size=hash_size, dct_size=dct_size)
    processed.position = position
    return processed

def process_frames(clip):
    processed_frames = []
    with multiprocessing.Pool(processes=15, initializer=process_frames_init) as pool:
        for processed in tqdm(pool.imap(process_frames_actor, enumerate(clip)), total=len(clip)):
            processed_frames.append(processed)
    
    for processed in processed_frames:
        processed.filename = clip.filename
    return processed_frames

# Nearest Matches
def find_nearest_matches_init(processed_source_frames, depth):
    global mp_data
    tree = VPTree(processed_source_frames, hamming_distance_processed) # HACK: tree might be too large to pickle
    mp_data = (tree, depth)

def find_nearest_matches_actor(processed):
    tree, depth = mp_data
    nearest_neighbors = tree.get_n_nearest_neighbors(processed, depth)
    nearest_neighbors = [element[1] for element in nearest_neighbors]
    processed.nearest_neighbors = nearest_neighbors
    processed.best_neighbor = nearest_neighbors[0]
    return processed

def find_nearest_matches(processed_source_frames, processed_edit_frames, depth=10):
    processed_edit_frames_new = []
    with multiprocessing.Pool(processes=15, initializer=find_nearest_matches_init, initargs=(processed_source_frames, depth)) as pool:
        for processed in tqdm(pool.imap(find_nearest_matches_actor, processed_edit_frames)):
            processed_edit_frames_new.append(processed)
    return processed_edit_frames_new

# Find closest near match
def find_better_nearest_matches(source_clips, edit_clip, processed_edit_frames):
    frames_to_scan = []
    for processed in processed_edit_frames:
        frames_to_scan.extend(processed.nearest_neighbors)
    
    frames_to_scan = sorted(list(set(frames_to_scan)), key=lambda processed: processed.position)
    for processed in tqdm(frames_to_scan):
        frame = source_clips[processed.filename][processed.position]
        adjusted_frame = crop_image_only_outside(frame, tol=40) # TODO: adjust?
        if adjusted_frame.size < 64 * 64 + 1:
            adjusted_frame = frame
        processed.hash = perceptual_hash(frame, hash_size=16, dct_size=64)

    for processed in tqdm(processed_edit_frames):
        frame = edit_clip[processed.position]
        adjusted_frame = crop_image_only_outside(frame, tol=40) # TODO: adjust?
        if adjusted_frame.size < 64 * 64 + 1:
            adjusted_frame = frame
        processed.hash = perceptual_hash(frame, hash_size=16, dct_size=64)

    tree = VPTree(frames_to_scan, hamming_distance_processed)
    for processed in tqdm(processed_edit_frames):
        processed.best_neighbor = tree.get_nearest_neighbor(processed)[1]

# Find closest matches
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

    #edit_filename = "../born.mp4"
    #source_filenames = ["../Halo1.mkv", "../Halo2.mkv"]

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
    processed_edit_frames = find_nearest_matches(processed_source_frames, processed_edit_frames)

    find_better_nearest_matches(source_clips, edit_clip, processed_edit_frames)

    with open("export4.pickle", "wb") as f:
        pickle.dump(processed_edit_frames, f)

    print("Displaying preview!")

    first = True
    for processed in processed_edit_frames:
        neighbor = processed.best_neighbor
        mat = source_clips[neighbor.filename][neighbor.position]
        cv2.imshow("Just the hams", mat)
        if first:
            cv2.waitKey(0)
            first = False
        cv2.waitKey(1)

    end_time = time.time()
    print("Ended the process at", end_time, "and it took", end_time - start_time)

    cv2.destroyAllWindows()

    print("Exporting data")

