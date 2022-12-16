import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class Camera(QThread):
    """Wraps cv2.VideoCapture and emits Qt signals with frames in RGB format.

    The :py:`run` function launches a loop that waits for new frames in
    the VideoCapture and emits them with a `new_frame` signal.  Calling
    :py:`stop` stops the loop and releases the camera.
    """

    frame_received = pyqtSignal(np.ndarray)
    # 定义信号为多维数组
    def __init__(self, video=0, parent=None, limit_fps=None):
        """Initialize Camera instance

        Args:
            video (int or string): ID of camera or video filename
            parent (QObject): parent object in Qt context
            limit_fps (float): force FPS limit, delay read if necessary.
        """

        QThread.__init__(self, parent=parent)
        self._cap = cv2.VideoCapture(video)
        # 打开笔记本内置摄像头
        self._running = False
        self._delay = 1 / limit_fps - 0.012 if limit_fps else np.nan
        # subtracting a roughly constant delay of 12ms TODO: better way?
        # np.nan will always evaluate to False in a comparison

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self._cap.read()
            # 读取视频返回视频是否结束的bool值和每一帧的图像
            last_time = time.perf_counter()
            # perf_counter()返回当前的计算机系统时间
            if not ret:
                # 当ret = 0时才会执行第一个选项（如果视频结束了）
                self._running = False
                raise RuntimeError("No frame received")
            else:
                self.frame_received.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 将信号frame的颜色通道从BGR转换为RGB再发送信号
                # 利用了rppg中的on_frame_received函数进行传参（传给frame_received）
            while (time.perf_counter() - last_time) < self._delay:
                # 计算程序运行时间
                time.sleep(0.001)
                # 控制程序执行的时间

    def stop(self):
        self._running = False
        time.sleep(0.1)
        self._cap.release()
        # 结束相机进程