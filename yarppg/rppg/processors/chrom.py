"""Chrominance-based rPPG method introduced by de Haan et al. [1]_


.. [1] de Haan, G., & Jeanne, V. (2013). Robust Pulse Rate From
   Chrominance-Based rPPG. IEEE Transactions on Biomedical Engineering,
   60(10), 2878–2886. https://doi.org/10.1109/TBME.2013.2266196
"""

import numpy as np

from .processor import Processor
import scipy.signal
import numpy as np
from yarppg.rppg.filters import DigitalFilter, get_butterworth_filter

# def fda(xs, Fstop1, Fstop2):
#     fs = 10
#     b, a = scipy.signal.butter(8, [Fstop1*2/fs, Fstop2*2/fs], 'bandpass')
#     xf = scipy.signal.filtfilt(b, a, xs)
#     return xf
def bdpass(xs, cutoff):
    fs = 30;
    lfilter = get_butterworth_filter(fs, cutoff, 'bandpass')
    xfilt = [lfilter(x) for x in xs]
    return xfilt

class ChromProcessor(Processor):

    def __init__(self, winsize=45, method="xovery"):
        Processor.__init__(self)

        self.winsize = winsize
        self.method = method

        self._xs, self._ys = [], []
        self.xmean, self.ymean = 0, 0
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi_pixels):
        self.n += 1
        r, g, b = self.spatial_pooling(roi_pixels, append_rgb=True)
        v = np.nan

        if self.method == "fixed":
            self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
            self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
            self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)
            rn = r / (self.rmean or 1.)
            gn = g / (self.gmean or 1.)
            bn = b / (self.bmean or 1.)
            self._xs.append(3*rn - 2*gn)
            self._ys.append(1.5*rn + gn - 1.5*bn)

            v = self._xs[-1] / (self._ys[-1] or 1.) - 1
        elif self.method == "xovery":
            self._xs.append(r - g)
            self._ys.append(0.5*r + 0.5*g - b)
            self.xmean = self.moving_average_update(self.xmean, self._xs, self.winsize)
            self.ymean = self.moving_average_update(self.ymean, self._ys, self.winsize)

            v = self.xmean / (self.ymean or 1) - 1
        elif self.method == "XsminaYs":
            Fstop1 = 40/60
            Fstop2 = 240 / 60
            cutoff = [Fstop1, Fstop2]
            self._rs = bdpass(self._rs, cutoff)
            self._gs = bdpass(self._gs, cutoff)
            self._bs = bdpass(self._bs, cutoff)
            # print("ghfgfdgfd", self._rs)
            self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
            self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
            self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)
            # print("和会", self.rmean)
            rn = r / (self.rmean or 1.)
            gn = g / (self.gmean or 1.)
            bn = b / (self.bmean or 1.)
            # print("hggh", rn)
            self._xs.append(3 * rn - 2 * gn)
            self._ys.append(1.5 * rn + gn - 1.5 * bn)
            # k = [0.7682, 0.5121, 0.3841]
            # rs = k[0]*rn
            # gs = k[1]*gn
            # bs = k[2]*bn
            # self._xs.append((rs-gs)/(k[0]-k[1]))
            # self._ys.append((rs+gs-2*bs)/(k[0]+k[1]-k[0]))

            d1 = np.std(self._xs)
            d2 = np.std(self._ys)
            a = d1/d2
            v = self._xs[-1] - a*self._ys[-1]
            # v = 3*(1-a/2)*self._rf[-1] - 2*(1+a/2)*self._gf[-1] + 3/2*a*self._bf[-1]
        return v

    def __str__(self):
        if self.name is None:
            return "ChromProcessor(winsize={},method={})".format(self.winsize,
                                                                 self.method)
        return self.name

