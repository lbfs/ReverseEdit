import json
import pprint
import subprocess
import os

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

        result = (min_match, max_match)

        ranges.append(result)

    return ranges

def parse_ffprobe_to_openshot(filename):
    command = f'ffprobe -v quiet -print_format json -show_streams {filename}'
    data = json.loads(subprocess.check_output(command).decode('UTF-8'))

    reader = {}
    reader['has_video'] = False
    reader['has_audio'] = False

   # print(data)

    for stream in data['streams']:

        if stream['codec_type'] == 'video':
            reader['has_video'] = True
            reader['height'] = int(stream['height'])
            reader['width'] = int(stream['width'])

            if 'metadata' not in reader:
                reader['metadata'] = {}

            # Figure out how to extract this better
            reader['metadata']['COMPATIBLE_BRANDS'] = 'iso6avc1mp41'
            reader['metadata']['MAJOR_BRAND'] = 'dash'
            reader['metadata']['MINOR_VERSION'] = '0'
            reader['metadata']['encoder'] = 'Lavf58.30.100'
            reader['video_bit_rate'] = 162251
            reader['video_length'] = "10824"
            reader["display_ratio"] = {
                    "den": 9,
                    "num": 16
            }
            reader["type"] = "FFmpegReader"
            reader["pixel_format"] = 0
            reader["pixel_ratio"] =  {
                    "den": 1,
                    "num": 1
            }

            reader['metadata']['HANDLER_NAME'] = stream['tags']['HANDLER_NAME']
            reader['metadata']['DURATION'] = stream['tags']['DURATION']
            reader['vcodec'] = stream['codec_name']
            # bit rate
            # video length
            # display ratio
            reader['video_stream_index'] = int(stream['index'])
            video_timebase_num, video_timebase_den = stream['time_base'].split('/')

            reader['video_timebase'] = {
                'num': int(video_timebase_num),
                'den': int(video_timebase_den)
            }

            reader['has_single_image'] = False
            reader['interlaced_frame'] = False

            reader['media_type'] = stream['codec_type']
            fps_num, fps_den = stream['r_frame_rate'].split('/')

            reader['fps'] = {
                'num': int(fps_num),
                'den': int(fps_den)
            }
        elif stream['codec_type'] == 'audio':
            reader['has_audio'] = True
            reader['audio_stream_index'] = int(stream['index'])
            reader['acodec'] = stream['codec_name']
            audio_timebase_num, audio_timebase_den = stream['time_base'].split('/')

            reader['audio_timebase'] = {
                'num': int(audio_timebase_num),
                'den': int(audio_timebase_den)
            }

            # Figure out how to extract this better
            reader["channels"] = stream['channels']
            reader["audio_bit_rate"] = 0
            reader["channel_layout"] = 3

            if 'metadata' not in reader:
                reader['metadata'] = {}

            if reader['has_video']:
                if stream['tags']['DURATION'] > reader['metadata']['DURATION']:
                    reader['metadata']['DURATION'] = stream['tags']['DURATION']

            if not reader['has_video']:
                reader['media_type'] = 'audio'

            if 'language' in stream['tags']:
                reader['metadata']['language'] = stream['tags']['language']

    reader['path'] = os.path.abspath(filename)
    reader['file_size'] = str(os.stat(filename).st_size)
    reader['title'] = os.path.basename(filename)
    return reader

def export_to_openshot(ranges, filenames):
    with open('templates/openshot.json', 'rt') as f:
        openshot = json.load(f)

    with open('templates/openshot-clip.json', 'rt') as f:
        openshot_clip = json.load(f)

    index = 1

    files_dict = {}
    for filename in filenames:
        result = parse_ffprobe_to_openshot(filename)
        result_id = str(index).zfill(10)
        result['id'] = result_id
        result['image'] = f'thumbnail/{result_id}.png'
        openshot['files'].append(result)
        files_dict[filename] = result
        index += 1

    for min_match, max_match in ranges:
        start = min_match.best_neighbor.timestamp
        end = max_match.best_neighbor.timestamp
        position = min_match.timestamp

        segment = openshot_clip.copy()
        segment['reader'] = files_dict[max_match.best_neighbor.filename].copy()
        segment['file_id'] = segment['reader']['id']
        del segment['reader']['id']
        del segment['reader']['image']

        segment['id'] = str(index).zfill(10)
        segment['start'] = start
        segment['position'] = position
        segment['end'] = end

        openshot['clips'].append(segment)
        index += 1

    with open('../../test2.osp', 'w') as json_file:
        json.dump(openshot, json_file, indent=4, sort_keys=True)

    return openshot

