import datetime

def split_frames_on_index_or_filename(matched_edit_frames, distance=5):
    indices = []

    for (x, y, i) in zip(matched_edit_frames, matched_edit_frames[1:], range(len(matched_edit_frames))):
        if distance < abs(x.best_neighbor.position - y.best_neighbor.position) or x.best_neighbor.filename != y.best_neighbor.filename:
            indices.append(i + 1)

    return [matched_edit_frames[start:end] for start, end in zip([0] + indices, indices + [len(matched_edit_frames)])]


def convert_splits_to_time_ranges(splits, invalid_if_less=30):
    ranges = []

    for entry in splits:
        if len(entry) < invalid_if_less:
            continue

        min_match = min(entry, key=lambda x: x.position)
        max_match = max(entry, key=lambda x: x.position)

        left = datetime.timedelta(seconds=min_match.timestamp)
        right = datetime.timedelta(seconds=max_match.timestamp)

        left_real = datetime.timedelta(
            seconds=min_match.best_neighbor.timestamp)
        right_real = datetime.timedelta(
            seconds=max_match.best_neighbor.timestamp)

        result = ((left, right), (left_real, right_real))

        ranges.append(result)

    return ranges

def export_to_kdenlive(splits):
    pass
