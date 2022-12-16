import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import scipy.signal


def bpm_from_inds(inds, ts):
    """Calculate heart rate (in beat/min) from indices and time vector

    Args:
        inds (`1d array-like`): indices of heart beats
        ts (`1d array-like`): time vector corresponding to indices

    Returns:
        float: heart rate in beats per minute (bpm)
    """

    if len(inds) < 2:
        return np.nan

    return 60. / np.mean(np.diff(ts[inds]))
# 计算数列ts[inds]中相邻两个元素之差，并取其均值

def get_sampling_rate(ts):
    # get_sampling_rate():获取取样率
    """Calculate sampling rate from time vector
    """
    return 1. / np.mean(np.diff(ts))
# diff计算相邻两个元素之差

def from_peaks(vs, ts, mindist=0.35):
    """Calculate heart rate by finding peaks in the given signal

    Args:
        vs (`1d array-like`): pulse wave signal
        ts (`1d array-like`): time vector corresponding to pulse signal
        mindist (float): minimum distance between peaks (in seconds)

    Returns:
        float: heart rate in beats per minute (bpm)
    """

    if len(ts) != len(vs) or len(ts) < 2:
        return np.nan
    f = get_sampling_rate(ts)
    peaks, _ = scipy.signal.find_peaks(vs, distance=int(f*mindist))
    # scipy.signal.find_peaks(x, height, threshold, distance, prominence, width, wlen, rel_height, plateau_size)
    # x：参数  height：所需的峰高  threshold：所需的峰值阈值  distance：相邻峰之间的样本中所需的最小水平距离（防止信号间的相互干扰）
    # prominence：所需的峰突出  width：样品所需的峰宽  wlen：用于计算峰值突出因此仅在参数之一时使用突出或者宽度给出
    # rel_height:用于计算峰宽  plateau_size:样品中峰的平顶大小
    return bpm_from_inds(peaks, ts)


def from_fft(vs, ts):
    """Calculate heart rate as most dominant frequency in pulse signal

    Args:
        vs (`1d array-like`): pulse wave signal
        ts (`1d array-like`): time vector corresponding to pulse signal

    Returns:
        float: heart rate in beats per minute (bpm)
    """

    f = get_sampling_rate(ts)
    vf = np.fft.fft(vs)
    # 进行快速傅里叶变换
    xf = np.linspace(0.0, f/2., len(vs)//2)
    # linspace(a, b, c):从[a, b]取c个数
    return 60 * xf[np.argmax(np.abs(vf[:len(vf)//2]))]
# np.argmax(f(x)):取x的最大值（求自变量的最大值）

class HRCalculator(QObject):
    new_hr = pyqtSignal(float)
    # 声明一个浮点类型参数的信号
    def __init__(self, parent=None, update_interval=30, winsize=300,
                 filt_fun=None, hr_fun=None):
        QObject.__init__(self, parent)

        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize
        self.filt_fun = filt_fun
        self.hr_fun = from_peaks
        if hr_fun is not None and callable(hr_fun):
            self.hr_fun = hr_fun

    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))
            if self.filt_fun is not None and callable(self.filt_fun):
                # callable:检查一个对象是否可调用
                vs = self.filt_fun(vs)
            self.new_hr.emit(self.hr_fun(vs, ts))
