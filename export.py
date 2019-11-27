def split_frames_on_index_or_filename(matched_edit_frames, distance = 5):
    indices = []

    for (x, y, i) in zip(matched_edit_frames, matched_edit_frames[1:], range(len(matched_edit_frames))):
        if distance < abs(x.best_neighbor.position - y.best_neighbor.position) or x.best_neighbor.filename != y.best_neighbor.filename:
            indices.append(i + 1)

    return [matched_edit_frames[start:end] for start, end in zip([0] + indices, indices + [len(matched_edit_frames)])]

def convert_splits_to_timestamps():
    pass

