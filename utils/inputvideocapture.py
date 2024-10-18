import cv2
import threading
import os
import time
from typing import Union, Tuple
from enum import Enum

class StreamType(Enum):
    WEBCAM = 0
    RTSP = 1
    FILE = 2

class InputVideoCapture:
    def __init__(self, src: Union[int, str], stream_type: int, resize: Union[Tuple[int, int], None] = None):
        self.resize = resize
        self.lock = threading.Lock()
        self.__retRTSP = False
        self.__lastFrameRTSP = None
        self.EOS = False

        if stream_type == StreamType.WEBCAM.value:
            self.cap = cv2.VideoCapture(int(src))
            self.__stream_type = StreamType.WEBCAM.value
        elif stream_type == StreamType.FILE.value:
            self.cap = cv2.VideoCapture(str(src))
            self.__stream_type = StreamType.FILE.value
            self.__frameId = 0
            self.__timeLastRead = None
        elif stream_type == StreamType.RTSP.value:
            self.cap = cv2.VideoCapture(str(src))
            self.__stream_type = StreamType.RTSP.value
            if not self.cap.isOpened():
                self.destroy()
                raise IOError(f"Error opening RTSP stream {src}")
            # Start the RTSP reading thread
            self.__reader = threading.Thread(target=self.__readRTSPFrame)
            self.__reader.start()

        if not self.cap.isOpened():
            self.destroy()
            raise IOError(f"Error opening video stream or file {src}")
    
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def read(self) -> Tuple[bool, cv2.Mat]:
        if self.isFile():
            return self.__getNextFrame()
        elif self.isWebcam():
            return self.__getNextFrame()
        else:
            with self.lock:
                return self.__retRTSP, self.__lastFrameRTSP
        
    def __readRTSPFrame(self):
        while not self.EOS:
            ret, frame = self.__getNextFrame()
            with self.lock:
                self.__retRTSP = ret
                self.__lastFrameRTSP = frame
            if not ret:
                time.sleep(0.01)

    def __getNextFrame(self) -> Tuple[bool, cv2.Mat]:
        ret, frame = self.cap.read()
        if ret and self.resize is not None:
            frame = cv2.resize(frame, self.resize)
        return ret, frame

    def destroy(self):
        self.EOS = True
        if self.__stream_type == StreamType.RTSP.value:
            self.__reader.join()
        self.cap.release()

    def reset(self):
        if not self.isFile():
            raise ReferenceError("Streams cannot be reset!")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def jumpTo(self, idframe: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idframe)

    def isWebcam(self) -> bool:
        return self.__stream_type == StreamType.WEBCAM.value
    
    def isFile(self) -> bool:
        return self.__stream_type == StreamType.FILE.value

    def getFPS(self) -> int:
        return self.fps

    def getTotalFrames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def getFrameSize(self) -> Tuple[int, int]:
        if self.resize is not None:
            return self.resize

        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
