import os
import shutil
import pickle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
from tqdm import tqdm
from gmpy2 import mpz, hamdist, pack

from vptree import VPTree
from clip import ClipReader
from image import ImageTool, HashedFrame
from export import split_frames_on_index_or_filename, convert_splits_to_time_ranges, export_to_openshot

mp_data = None


def apply_hash_initalizer(hash_function, hash_args, hash_window_size, tolerance):
    global mp_data
    mp_data = (hash_function, hash_args,
               hash_window_size, tolerance)


def apply_hash_actor(frame):
    global mp_data
    hash_function, hash_args, hash_window_size, tolerance = mp_data

    data = ImageTool.crop_image_only_outside(
        frame.frame, tolerance, hash_window_size, hash_window_size)

    hashed_frame = HashedFrame()
    hashed_frame.hash = hash_function(data, *hash_args)
    hashed_frame.hash = pack(hashed_frame.hash.astype(int).flatten().tolist(), 1)
    hashed_frame.position = frame.position
    hashed_frame.filename = frame.filename
    return hashed_frame

def apply_hash(iterator, hash_function, hash_args, hash_window_size, tolerance=40, iterator_length=None):
    initargs = (hash_function, hash_args, hash_window_size, tolerance,)
    process_count = multiprocessing.cpu_count() - 1 or 1
    with ThreadPool(processes=process_count, initializer=apply_hash_initalizer, initargs=initargs) as pool:
        iterator = tqdm(pool.imap(apply_hash_actor,
                                  iterator), total=iterator_length)
        frames = [hashed_frame for hashed_frame in iterator]

    return frames

def apply_timestamps(frames, clip):
    for frame in frames:
        frame.timestamp = float(frame.position) / clip.fps

def find_nearest_matches_initalizer(frames, depth):
    global mp_data
    # HACK: We should only need to construct the tree on the main process and then send it
    # However, some trees might be too large to recurse, which is a significant problem.
    # Thankfully, we can easily just create the tree here and there are no issues,
    # except higher than normal CPU usage.
    tree = VPTree(frames, HashedFrame.compute_distance)
    mp_data = (tree, depth)


def find_nearest_matches_actor(frame):
    tree, depth = mp_data
    nearest_neighbors = tree.get_n_nearest_neighbors(frame, depth)
    nearest_neighbors = [element[1] for element in nearest_neighbors]
    frame.nearest_neighbors = nearest_neighbors
    frame.best_neighbor = nearest_neighbors[0]
    return frame


def find_nearest_matches(source_frames, edit_frames, depth=1):
    process_count = multiprocessing.cpu_count() - 1 or 1
    with multiprocessing.Pool(processes=process_count, initializer=find_nearest_matches_initalizer, initargs=(source_frames, depth)) as pool:
        iterator = tqdm(pool.imap(find_nearest_matches_actor,
                                  edit_frames), total=len(edit_frames))
        frames = [frame for frame in iterator]
    return frames

def build(edit_filename, source_filenames):
    print("Phase 0: Hashing and Cropping")

    hash_function = ImageTool.perceptual_hash
    hash_size = 64
    hash_args = (hash_size, hash_size * 4)
    hash_window_size = hash_size * 4 * hash_size * 4

    source_clips = {}
    source_frames = []
    for filename in source_filenames:
        print("Processing", filename)
        source_clips[filename] = ClipReader(filename)
        frames = apply_hash(source_clips[filename], hash_function=hash_function, hash_args=hash_args, hash_window_size=hash_window_size, iterator_length=len(source_clips[filename]))
        apply_timestamps(frames, source_clips[filename])
        source_frames.extend(frames)

    print("Processing", edit_filename)
    edit_clip = ClipReader(edit_filename)
    edit_frames = apply_hash(edit_clip, hash_function=hash_function, hash_args=hash_args,
                             hash_window_size=hash_window_size, iterator_length=len(edit_clip))
    apply_timestamps(edit_frames, edit_clip)

    print("Phase 1: Finding Nearest Matches")
    matched_edit_frames = find_nearest_matches(source_frames, edit_frames, depth=1)

    print("Phase 2: Export Debug Data")
    with open("export.pickle", "wb") as f:
        pickle.dump(matched_edit_frames, f)

    print("Phase 3: Export Frames")
    splits = split_frames_on_index_or_filename(matched_edit_frames, distance = 15)
    ranges = convert_splits_to_time_ranges(splits, 30)
    result = export_to_openshot(ranges, source_filenames)

if __name__ == "__main__":
    edit_filename = "../time.mkv"
    source_filenames = ["../Halo3.mkv"]

    build(edit_filename, source_filenames)