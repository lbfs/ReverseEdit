import json

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

        result = ((min_match.timestamp, max_match.timestamp), (min_match.best_neighbor.timestamp, max_match.best_neighbor.timestamp))

        ranges.append(result)

    return ranges


def export_to_openshot(ranges):
    with open("openshot.json", "rt") as osf:
        openshot = json.load(osf)

    with open("clip.json", "rt") as cosf:
        clip = json.load(cosf)

    for index, t_range in enumerate(ranges):
        start = t_range[1][0]
        end = t_range[1][1]
        position = t_range[0][0]

        segment = clip.copy()
        segment["id"] = str(index).zfill(10)
        segment["start"] = start
        segment["position"] = position
        segment["end"] = end

        openshot["clips"].append(segment)

    with open('../../test2.osp', 'w') as json_file:
        json.dump(openshot, json_file, indent=4, sort_keys=True)

    return openshot

