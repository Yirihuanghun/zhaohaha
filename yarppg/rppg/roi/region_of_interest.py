import functools
import sys
from multiprocessing.sharedctypes import Value
import cv2
import numpy as np
import dlib
def pixelate(img, xywh, blur):
    if blur > 0:
        x, y, w, h = xywh
        slicex = slice(x, x+w)
        slicey = slice(y, y+h)

        tmp = cv2.resize(img[slicey, slicex], (w//blur, h//blur),
                         interpolation=cv2.INTER_LINEAR)
        # 以INTER_LINEAR的方法缩小图像尺寸
        img[slicey, slicex] = cv2.resize(tmp, (w, h),
                                         interpolation=cv2.INTER_NEAREST)

@functools.lru_cache(maxsize=2)
def get_default_bgmask(w, h):
    mask = np.zeros((h, w), dtype="uint8")
    # dtype：定义数据类型
    cv2.rectangle(mask, (0, 0), (w, 5), 255, -1)

    return mask


class RegionOfInterest:
    def __init__(self, base_img, mask=None, bgmask=None, facerect=None):
        self.rawimg = base_img

        self._mask = mask
        self._rectangle = None
        self._empty = True
        self._rectangular = False
        self._contours = None
        self._bgmask = bgmask
        self._facerect = facerect

        if mask is not None:
            self._rectangle = cv2.boundingRect(mask)
            # boundingRect:计算轮廓的垂直边界最小矩形
            self._empty = (self._rectangle[2] == 0 or self._rectangle[3] == 0)

    @classmethod
    def from_rectangle(cls, base_img, p1, p2, **kwargs):
        # https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
        # cls：和self用法类似，但不用实例化（即不用赋予一个对象），直接类名.方法名()来调用
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        # 建立行数为图像的高，列数为图像的宽的掩模
        cv2.rectangle(mask, p1, p2, 255, cv2.FILLED)
        # 指定mask为画板，p1为绘制边框的左上角顶点的坐标，p2为绘制边框右下角顶点的坐标，255为颜色设置
        # cv2.FILLED:全连接线
        roi = RegionOfInterest(base_img, mask=mask, **kwargs)
        # RegionOfInterest:截取部分影像
        roi._rectangular = True

        return roi

    @classmethod
    def from_contour(cls, base_img, pointlist, **kwargs):
        # pointlist with shape nx2
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        contours = pointlist.reshape((1, -1, 1, 2))
        # 对目标数组pointlist取数据重新布局
        cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)
        # drawContours：画出图像的轮廓
        # 在mask上对contours轮廓圈出的区域进行填充
        roi = RegionOfInterest(base_img, mask, **kwargs)
        # 利用RegionOfInterest函数在base_img图像上剪切出在mask图像上被填充的区域
        # **kwargs：关键字参数，用于函数定义，可以将不定数量的参数传递给一个函数
        roi._contours = contours
        return roi

    def draw_roi(self, img, color=(255, 0, 0), thickness=3):
        if self.is_empty():
            return

        if self.is_rectangular():
            p1, p2 = self.get_bounding_box(as_corners=True)
            # 获得边界框的左上角坐标和右下角坐标
            cv2.rectangle(img, p1, p2, color, thickness)
            # 在img图像上画出矩形框
        else:
            cv2.drawContours(img, self._contours, 0, color=color,
                             thickness=thickness)
            # drawContours:在图像上进行轮廓绘制，self._contours为轮廓本身（是一个list），第三个参数指定绘制轮廓list中的哪条轮廓，为-1则全部绘制
    def pixelate_face(self, img, blursize):
        if not self.is_empty():
            xywh = self._rectangle if self._facerect is None else self._facerect
            pixelate(img, xywh, blursize)

    def is_rectangular(self):
        return self._rectangular

    def is_empty(self):
        return self._empty

    def get_bounding_box(self, as_corners=False):
        """Bounding box specified as (x, y, w, h) or min/max corners
        """
        if as_corners:
            x, y, w, h = self._rectangle
            return (x, y), (x+w, y+h)
        return self._rectangle

    def get_mean_rgb(self, background=False):
        mask = self._mask
        if background:
            if self._bgmask is None:
                raise ValueError("Background mask is not specified")
            # 未指定背景掩模
            mask = self._bgmask

        r, g, b, a = cv2.mean(self.rawimg, mask)
        # rawimg和mask为同样大小的矩阵，计算rawimg中所有元素的均值，为0的地方不计算
        # rawimg = base_img
        # 获取图像四通道各自的均值（a为透明度）
        return r, g, b

    def __str__(self):
        if self.is_empty():
            return "RegionOfInterest(empty)"
        if self.is_rectangular():
            return f"RegionOfInterest(rect={self._rectangle})"

        return f"RegionOfInterest(masked within bb={self._rectangle})"
