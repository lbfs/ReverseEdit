import os
import cv2


class Frame:
    __slots__ = ["frame", "filename", "position"]

    def __init__(self, frame, filename, position):
        self.frame = frame
        self.filename = filename
        self.position = position


class ClipReader:
    """ A class to interact with a video object with easy seek and iteration functionality. """
    __slots__ = ["filename", "pos", "capture",
                 "fps", "height", "width", "count"]

    def __init__(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

        self.filename = filename
        self.pos = 0
        self.capture = cv2.VideoCapture(filename)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        """
        Iterate through all of the frames from starting position of a video to the end of the video.
        """
        pos = 0
        while True:
            if self.pos != pos:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
                self.pos = pos

            ret, frame = self.capture.read()

            if ret:
                yield Frame(frame, self.filename, self.pos)
                pos += 1
                self.pos = pos
            else:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.pos = 0
                break

    def __getitem__(self, key):
        """
        Requests the desired frame at the specified index.
        Accessing sequential frames does not require additional seeks.
        """
        if not isinstance(key, int):
            raise IndexError(
                f"ClipReader requires an integer index not {type(key)}")

        # if key < 0 or key >= len(self):
        #    raise IndexError(f"ClipReader index is out of range.")

        if key == self.pos + 1:
            ret, frame = self.capture.read()
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, key)
            ret, frame = self.capture.read()

        self.pos = key + 1

        if not ret:
            return None

        return Frame(frame, self.filename, key)

    def __len__(self):
        """
        Returns the estimated number of frames present in the Clip
        """
        return self.count
