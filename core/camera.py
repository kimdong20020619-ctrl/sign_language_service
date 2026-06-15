import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, camera_index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 30):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"카메라 {self.camera_index}를 열 수 없습니다.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        interval_ms = max(1, 1000 // self.fps)

        while self._running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # 거울 모드
                self.frame_ready.emit(frame)
            else:
                self.error_occurred.emit("프레임을 읽을 수 없습니다.")
                break
            self.msleep(interval_ms)

        self.cap.release()
        self.cap = None

    def stop(self):
        self._running = False
        self.wait(3000)

    def set_camera(self, index: int):
        was_running = self._running
        if was_running:
            self.stop()
        self.camera_index = index
        if was_running:
            self.start()

    @property
    def is_active(self) -> bool:
        return self._running and self.isRunning()
