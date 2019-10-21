import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import os
import shutil
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from vptree import VPTree
from clip import ClipReader
from image import ImageTool, HashedFrame
from gmpy2 import mpz, hamdist, pack

mp_data = None


def apply_hash_initalizer(hash_function, hash_settings, hash_window_size, tolerance):
    global mp_data
    mp_data = (hash_function, hash_settings,
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


def debug_export(edit_clip, source_clips, edit_frames):
    if os.path.exists("../debug_export"):
        shutil.rmtree("../debug_export")

    os.mkdir("../debug_export")
    for index, frame in tqdm(enumerate(edit_frames), total=len(edit_frames)):
        neighbor = frame.best_neighbor
        edit_frame = edit_clip[frame.position].frame
        source_frame = source_clips[neighbor.filename][neighbor.position].frame
        source_frame = cv2.resize(
            source_frame, (edit_clip.width, edit_clip.height))

        distance = HashedFrame.compute_distance(frame, neighbor)
        frame_to_write = np.concatenate((edit_frame, source_frame), axis=1)

        cv2.putText(frame_to_write, f"{distance:0.2f}", (
            10, frame_to_write.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        cv2.imwrite(f"../debug_export/{index:05d}.png", frame_to_write)


def debug_build(edit_filename, source_filenames):
    with open("export.pickle", "rb") as f:
        edit_frames = pickle.load(f)

    source_clips = {}
    for filename in source_filenames:
        source_clips[filename] = ClipReader(filename)

    edit_clip = ClipReader(edit_filename)
    debug_export(edit_clip, source_clips, edit_frames)
    return edit_frames

def frames_from_multiclip(frames, clips):
    # Currently unused
    frames = sorted(frames) #Sorting the frames in order helps with reducing seek errors.
    for frame in frames:
        yield clips[frame.filename][frame.position]

def build(edit_filename, source_filenames):
    print("Phase 0: Hashing and Cropping")

    hash_function = ImageTool.perceptual_hash
    hash_size = 32
    hash_args = (hash_size, hash_size * 4)
    hash_window_size = hash_size * 4 * hash_size * 4

    source_clips = {}
    source_frames = []
    for filename in source_filenames:
        print("Processing", filename)
        source_clips[filename] = ClipReader(filename)
        source_frames.extend(apply_hash(source_clips[filename], hash_function=hash_function,
                                        hash_args=hash_args, hash_window_size=hash_window_size, iterator_length=len(source_clips[filename])))

    print("Processing", edit_filename)
    edit_clip = ClipReader(edit_filename)
    edit_frames = apply_hash(edit_clip, hash_function=hash_function, hash_args=hash_args,
                             hash_window_size=hash_window_size, iterator_length=len(edit_clip))

    print("Phase 1: Finding Nearest Matches")
    matched_edit_frames = find_nearest_matches(source_frames, edit_frames, depth=1)

    print("Phase 2: Export Debug Data")
    with open("export.pickle", "wb") as f:
        pickle.dump(matched_edit_frames, f)

    print("Phase 3: Export Frames")
    debug_export(edit_clip, source_clips, matched_edit_frames)
    return matched_edit_frames


if __name__ == "__main__":
    edit_filename = "../hawkling.mkv"
    source_filenames = ["../ark.mkv"]
    #edit_filename = "../Forever.mkv"
    #source_filenames = ["../Halo3.mkv", "../Halo2.mkv",
    #                    "../Wars.mkv", "../Starry.mkv", "../ODST.mkv", "../E3.mkv"]

    debug = False
    if debug:
        matched_edit_frames = debug_build(edit_filename, source_filenames)
    else:
        matched_edit_frames = build(edit_filename, source_filenames)

    cv2.destroyAllWindows()
