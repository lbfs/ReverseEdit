import os
import cv2

class ClipReader:
    """ A class to interact with a video object with easy seek and iteration functionality. """
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

        self.filename = filename
        self.pos = 0
        self.capture = cv2.VideoCapture(filename)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self._count_frames()

    def _count_frames(self):
        """ 
        Count the amount of frames in the VideoCapture. 
        Accessing cv2.CAP_PROP_FRAME_COUNT is unreliable.
        Should not be called as this result will be computed into the self.count variable on initalization.
        A small run-time cost is incured as the entire video needs to be seeked through. 
        """
        count = 0
        for element in iter(self):
            count += 1
        self.count = count

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
                yield frame
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
            raise IndexError(f"ClipReader requires an integer index not {type(key)}")

        if key < 0 or key >= len(self):
            raise IndexError(f"ClipReader index is out of range.")

        if key == self.pos + 1:
            _, frame = self.capture.read()
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, key)
            _, frame = self.capture.read()

        self.pos = key
        return frame

    def __len__(self):
        """
        Returns the number of frames present in the Clip
        """
        return self.count
