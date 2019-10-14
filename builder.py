import time
import pickle
import os
import shutil
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

    def __lt__(self, other):
        return self.position < other.position

def hamming_distance_processed(u, v):
    return hamming_distance(u.hash, v.hash)

# Phase 0: Crop and hash frames
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
    adjusted_frame = crop_image_only_outside(frame, tol=30)

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

# Phase 1: Find Nearest Matches
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

# Phase 2: Find Better Nearest Matches
def find_better_nearest_matches_init(frames_to_scan):
    global mp_data
    tree = VPTree(frames_to_scan, hamming_distance_processed) # HACK: tree might be too large to pickle
    mp_data = tree

def find_better_nearest_matches_actor(processed):
    processed.best_neighbor = mp_data.get_nearest_neighbor(processed)[1]
    return processed

def find_better_nearest_matches(source_clips, edit_clip, processed_edit_frames):
    hash_size = 16
    dct_size = hash_size * 4
    window_size = dct_size * dct_size
    nearest_neighbors = []
    # TODO: Parallelize
    for processed in tqdm(processed_edit_frames):
        frame = edit_clip[processed.position]
        adjusted_frame = crop_image_only_outside(frame, tol=30) # TODO: adjust?
        if adjusted_frame.size < window_size:
            adjusted_frame = frame
        processed.hash = perceptual_hash(frame, hash_size=hash_size, dct_size=dct_size)
        nearest_neighbors.extend(processed.nearest_neighbors)

    # We order and remove duplicates, also helps (does not prevent) with preventing seek errors by keeping everything linear.
    nearest_neighbors = sorted(list(set(nearest_neighbors)))
    # TODO: Parallelize
    for neighbor in tqdm(nearest_neighbors):
        frame = source_clips[neighbor.filename][neighbor.position]
        adjusted_frame = crop_image_only_outside(frame, tol=30) # TODO: adjust?
        if adjusted_frame.size < window_size:
            adjusted_frame = frame
        neighbor.hash = perceptual_hash(frame, hash_size=hash_size, dct_size=dct_size)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1 or 1, initializer=find_better_nearest_matches_init, initargs=(nearest_neighbors,)) as pool:
        for index, processed in tqdm(enumerate(pool.imap(find_better_nearest_matches_actor, processed_edit_frames)), total=len(processed_edit_frames)):
            processed_edit_frames[index] = processed

def debug_export(edit_clip, source_clips, processed_edit_frames):
    if os.path.exists("../debug_export"):
        shutil.rmtree("../debug_export")

    os.mkdir("../debug_export")    
    for index, processed in enumerate(processed_edit_frames):
        neighbor = processed.best_neighbor
        edit_frame = edit_clip[processed.position]
        source_frame = source_clips[neighbor.filename][neighbor.position]
        source_frame = cv2.resize(source_frame, (edit_clip.width, edit_clip.height))
        cv2.imwrite(f"../debug_export/{index:05d}.png", np.concatenate((edit_frame, source_frame), axis=1))

def skip_debug_export(edit_filename, source_filenames):
    with open("export.pickle", "rb") as f:
        processed_edit_frames = pickle.load(f)

    source_clips = {}
    for filename in source_filenames:
        source_clips[filename] = ClipReader(filename)
    
    edit_clip = ClipReader(edit_filename)
    debug_export(edit_clip, source_clips, processed_edit_frames)

def build(edit_filename, source_filenames):
    print("Phase 0: Hashing and Cropping")
    print("Processing", edit_filename)
    edit_clip = ClipReader(edit_filename)
    processed_edit_frames = process_frames(edit_clip)

    source_clips = {}
    processed_source_frames = []
    for filename in source_filenames:
        print("Processing", filename)
        source_clips[filename] = ClipReader(filename)
        processed_source_frames.extend(process_frames(source_clips[filename]))

    print("Phase 1: Finding Nearest Matches")
    processed_edit_frames = find_nearest_matches(processed_source_frames, processed_edit_frames)

    print("Phase 2: Finding Best Matches")
    find_better_nearest_matches(source_clips, edit_clip, processed_edit_frames)

    print("Phase 3: Drop Mostly Solid Color Frames")
    
    print("Phase 4: Drop Frames Below Comparison Threshold")
    
    print("Phase 5: Sort Frames By Chunked Segment (Maybe?)")

    print("Phase 6: Export Debug Data")
    with open("export.pickle", "wb") as f:
        pickle.dump(processed_edit_frames, f)

    print("Phase 6: Export Frames")
    debug_export(edit_clip, source_clips, processed_edit_frames)

    return processed_edit_frames

if __name__ == "__main__":
    edit_filename = "../hawkling.mkv"
    source_filenames = ["../ark.mkv"]
    #edit_filename = "../Forever.mkv"
    #source_filenames = ["../Halo3_720.mp4", "../Halo2.mkv", "../Wars.mkv", "../Starry.mkv", "../ODST.mkv", "../E3.mkv"]

    debug = 0
    if debug:
        debug_export(edit_filename, source_filenames)
    else:
        processed_edit_frames = build(edit_filename, source_filenames)

        with open("export.pickle", "wb") as f:
            pickle.dump(processed_edit_frames, f)

    cv2.destroyAllWindows()
