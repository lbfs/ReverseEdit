import os
import argparse
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm
from gmpy2 import mpz, hamdist, pack

from vptree import VPTree
from clip import ClipReader
from image import ImageTool, HashedFrame
from export import split_frames_on_index_or_filename, convert_splits_to_time_ranges, export_to_openshot

mp_data = None


def apply_hash_initalizer(hash_function, hash_args, hash_window_size, tolerance):
    """
    Sets up the hashing and cropping data inside the global mp_data variable.
    """
    global mp_data
    mp_data = (hash_function, hash_args,
               hash_window_size, tolerance)


def apply_hash_actor(frame):
    """
    Reads the data stored by apply_hash_initalizer and applies cropping and hashing to
    each of the frames passed to this function. Returns a HashedFrame() with the results.
    """
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
    """
    Iterates through each frame in a supplied video and applies the apply_hash_actor in parallel.
    Returns a set of HashedFrames() supplying video information.
    """
    initargs = (hash_function, hash_args, hash_window_size, tolerance,)
    process_count = multiprocessing.cpu_count() - 1 or 1
    with ThreadPool(processes=process_count, initializer=apply_hash_initalizer, initargs=initargs) as pool:
        iterator = tqdm(pool.imap(apply_hash_actor,
                                  iterator), total=iterator_length)
        frames = [hashed_frame for hashed_frame in iterator]

    return frames

def apply_timestamps(frames, clip):
    """ Convert each frame index into a frame timestamp in seconds """
    for frame in frames:
        frame.timestamp = float(frame.position) / clip.fps

def find_nearest_matches_initalizer(frames, depth):
    """ Build a VPTree of HashedFrames for parallel lookup of edit frames. """
    global mp_data
    # HACK: We should only need to construct the tree on the main process and then send it
    # However, some trees might be too large to recurse, which is a significant problem.
    # Thankfully, we can easily just create the tree here and there are no issues,
    # except higher than normal CPU usage.
    tree = VPTree(frames, HashedFrame.compute_distance)
    mp_data = (tree, depth)


def find_nearest_matches_actor(frame):
    """ Scan the VPTree for the closest matching frame and store it in the HashedFrame. """
    tree, depth = mp_data
    nearest_neighbors = tree.get_n_nearest_neighbors(frame, depth)
    nearest_neighbors = [element[1] for element in nearest_neighbors]
    frame.nearest_neighbors = nearest_neighbors
    frame.best_neighbor = nearest_neighbors[0]
    return frame


def find_nearest_matches(source_frames, edit_frames, depth=1):
    """ Find the nearest matches to each frame in the edit video. """
    process_count = multiprocessing.cpu_count() - 1 or 1
    with multiprocessing.Pool(processes=process_count, initializer=find_nearest_matches_initalizer, initargs=(source_frames, depth)) as pool:
        iterator = tqdm(pool.imap(find_nearest_matches_actor,
                                  edit_frames), total=len(edit_frames))
        frames = [frame for frame in iterator]
    return frames

def build(edit_filename, source_filenames, hash_size, split_distance, invalid_less, export_path, edit_tolerance, source_tolerance):
    """ Build the Openshot Video Project File out of the provided settings. """
    print("Phase 0: Hashing and Cropping")

    hash_function = ImageTool.perceptual_hash
    hash_args = (hash_size, hash_size * 4)
    hash_window_size = hash_size * 4 * hash_size * 4

    source_clips = {}
    source_frames = []
    for filename in source_filenames:
        print("Processing", filename)
        source_clips[filename] = ClipReader(filename)
        frames = apply_hash(source_clips[filename], hash_function=hash_function, hash_args=hash_args, hash_window_size=hash_window_size, tolerance=source_tolerance, iterator_length=len(source_clips[filename]))
        apply_timestamps(frames, source_clips[filename])
        source_frames.extend(frames)

    print("Processing", edit_filename)
    edit_clip = ClipReader(edit_filename)
    edit_frames = apply_hash(edit_clip, hash_function=hash_function, hash_args=hash_args,
                             hash_window_size=hash_window_size, iterator_length=len(edit_clip), tolerance=edit_tolerance)
    apply_timestamps(edit_frames, edit_clip)

    print("Phase 1: Finding Nearest Matches")
    matched_edit_frames = find_nearest_matches(source_frames, edit_frames, depth=1)

    print("Phase 2: Exporting to File")
    splits = split_frames_on_index_or_filename(matched_edit_frames, distance = split_distance)
    ranges = convert_splits_to_time_ranges(splits, invalid_less)
    result = export_to_openshot(export_path, ranges, source_filenames)

    print(f"Exported to {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recreate a video project file from source footage and an already existing edited video.')

    parser.add_argument('--source_filename', action='append', dest="source_filenames")
    parser.add_argument('--edit_filename', action='store', dest="edit_filename")
    parser.add_argument('--hash_size', action='store', dest='hash_size', default=32, type=int)
    parser.add_argument('--edit_tolerance', action='store', dest='edit_tolerance', default=40, type=int)
    parser.add_argument('--source_tolerance', action='store', dest='source_tolerance', default=40, type=int)
    parser.add_argument('--split_distance', action='store', dest='split_distance', default=15, type=int)
    parser.add_argument('--invalid_less', action='store', dest='invalid_less', default=30, type=int)
    parser.add_argument('--export_path', action='store', dest='export_path', default="export.osp")

    args = parser.parse_args()

    if args.source_filenames == None:
        print("No source videos were provided. Please use --source_filename [path].")
        exit(1)

    if args.edit_filename == None:
        print("No edit video was provided. Please provide with --edit_filename [path]")
        exit(1)

    for source_filename in args.source_filenames:
        if not os.path.exists(source_filename):
            print(f"Source filename: {source_filename} does not exist. Aborting.")
            exit(1)

    if not os.path.exists(args.edit_filename):
        print(f"Edit filename: {args.edit_filename} does not exist. Aborting.")

    if args.hash_size % 2 != 0 or args.hash_size < 8:
        print("Hash size is not a power of 2 or greater than 8. Aborting")
        exit(1)

    print("---------------------------------------")
    print("Source Filenames:", args.source_filenames)
    print("Edit Filenames:", args.edit_filename)
    print("Edit Tolerance:", args.edit_tolerance)
    print("Source Tolerance:", args.source_tolerance)
    print("Hash Size:", args.hash_size)
    print("Split distance:", args.split_distance)
    print("Invalidate Frame Distance:", args.invalid_less)
    print("Export Filename:", args.export_path)
    print("---------------------------------------")

    build(args.edit_filename, args.source_filenames, args.hash_size, args.split_distance, args.invalid_less, args.export_path, args.edit_tolerance, args.source_tolerance)