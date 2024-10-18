import cv2
import numpy as np
from typing import Tuple

class OutputVideoWriter:

    def __init__(self, output_path: str, fps: int, size: Tuple[int, int]):
        self.__outputVideo = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                            fps,
                            size
                        )
        self.__count = 0
        self.size = size

    def __del__(self):
        if self.__outputVideo:
            self.__outputVideo.release()

    def write(self, frame: np.ndarray) -> None:
        frame = cv2.resize(frame,self.size)
        self.__outputVideo.write(frame)
        self.__count += 1

    def count(self) -> int:
        return self.__count
