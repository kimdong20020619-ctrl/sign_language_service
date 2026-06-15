from queue import Empty, Queue

import numpy as np
from PySide6.QtCore import QThread, Signal


class MediaPipeWorker(QThread):
    """MediaPipe 처리를 별도 스레드에서 실행.

    카메라 프레임을 큐에 받아 처리하고, 결과를 frame_processed 시그널로 전송.
    큐가 꽉 차면 오래된 프레임을 버려서 지연 누적을 방지.
    """

    frame_processed = Signal(np.ndarray, object)  # annotated, frame_data

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._queue: Queue = Queue(maxsize=2)
        self._running = False

    def submit_frame(self, frame: np.ndarray) -> None:
        """카메라 스레드에서 호출 — 큐가 가득 차면 오래된 프레임 드롭."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except Exception:
            pass

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                frame = self._queue.get(timeout=0.05)
                annotated, frame_data = self._engine.process_frame(frame)
                self.frame_processed.emit(annotated, frame_data)
            except Empty:
                continue
            except Exception as exc:
                print(f"[MediaPipeWorker] {exc}")

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
