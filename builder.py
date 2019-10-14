import time
import pickle
import multiprocessing
import cv2
import numpy as np
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

def hamming_distance_processed(u, v):
    return hamming_distance(u.hash, v.hash)

# Crop and hash frames
def process_frames_init():
    global mp_data
    hash_size = 8
    dct_size = hash_size * 4
    window_size = dct_size * dct_size
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
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1 or 1, initializer=process_frames_init) as pool:
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
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1 or 1, initializer=find_nearest_matches_init, initargs=(processed_source_frames, depth)) as pool:
        for processed in tqdm(pool.imap(find_nearest_matches_actor, processed_edit_frames), total=len(processed_edit_frames)):
            processed_edit_frames_new.append(processed)
    return processed_edit_frames_new

# Find closest near match
def find_better_nearest_matches_init(frames_to_scan):
    global mp_data
    tree = VPTree(frames_to_scan, hamming_distance_processed) # HACK: tree might be too large to pickle
    mp_data = tree

def find_better_nearest_matches_actor(processed):
    processed.best_neighbor = mp_data.get_nearest_neighbor(processed)[1]
    return processed

def find_better_nearest_matches(source_clips, edit_clip, processed_edit_frames):
    frames_to_scan = []
    for processed in processed_edit_frames:
        frames_to_scan.extend(processed.nearest_neighbors)
    
    frames_to_scan = sorted(list(set(frames_to_scan)), key=lambda processed: processed.position)

    dct_total_size = 64 * 64
    for processed in tqdm(frames_to_scan):
        frame = source_clips[processed.filename][processed.position]
        adjusted_frame = crop_image_only_outside(frame, tol=40) # TODO: adjust?
        if adjusted_frame.size < dct_total_size:
            adjusted_frame = frame
        processed.hash = perceptual_hash(frame, hash_size=16, dct_size=64)

    for processed in tqdm(processed_edit_frames):
        frame = edit_clip[processed.position]
        adjusted_frame = crop_image_only_outside(frame, tol=40) # TODO: adjust?
        if adjusted_frame.size < dct_total_size:
            adjusted_frame = frame
        processed.hash = perceptual_hash(frame, hash_size=16, dct_size=64)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1 or 1, initializer=find_better_nearest_matches_init, initargs=(frames_to_scan,)) as pool:
        for index, processed in tqdm(enumerate(pool.imap(find_better_nearest_matches_actor, processed_edit_frames)), total=len(processed_edit_frames)):
            processed_edit_frames[index] = processed

def build(edit_filename, source_filenames):
    start_time = time.time()

    print("Starting Phase 1")
    print("Processing", edit_filename)
    edit_clip = ClipReader(edit_filename)
    processed_edit_frames = process_frames(edit_clip)

    source_clips = {}
    processed_source_frames = []
    for filename in source_filenames:
        print("Processing", filename)
        source_clips[filename] = ClipReader(filename)
        processed_source_frames.extend(process_frames(source_clips[filename]))

    print("Calculating nearest matches")
    processed_edit_frames = find_nearest_matches(processed_source_frames, processed_edit_frames)

    print("Starting Phase 2")
    find_better_nearest_matches(source_clips, edit_clip, processed_edit_frames)

    end_time = time.time()
    print("Recreation attempt took", end_time - start_time, "seconds.")
    
    return processed_edit_frames

def debug_export(edit_filename, source_filenames):
    source_clips = {}
    for filename in source_filenames:
        source_clips[filename] = ClipReader(filename)
    
    edit_clip = ClipReader(edit_filename)

    with open("export.pickle", "rb") as f:
        processed_edit_frames = pickle.load(f)
    
    print("Displaying preview!")

    first = True
    for index, processed in enumerate(processed_edit_frames):
        neighbor = processed.best_neighbor
        edit_frame = edit_clip[processed.position]

        source_frame = source_clips[neighbor.filename][neighbor.position]
        source_frame = cv2.resize(source_frame, (edit_clip.width, edit_clip.height))

        cv2.imwrite(f"../debug_export/{index:05d}.png", np.concatenate((edit_frame, source_frame), axis=1))

if __name__ == "__main__":
    edit_filename = "../Forever.mkv"
    source_filenames = ["../Halo3_720.mp4", "../Halo2.mkv", "../Wars.mkv", "../Starry.mkv", "../ODST.mkv", "../E3.mkv"]

    debug = 1
    if debug:
        debug_export(edit_filename, source_filenames)
    else:
        processed_edit_frames = build(edit_filename, source_filenames)

        with open("export.pickle", "wb") as f:
            pickle.dump(processed_edit_frames, f)

    cv2.destroyAllWindows()
