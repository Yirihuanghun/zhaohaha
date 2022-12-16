import numpy as np
import scipy.signal


def get_butterworth_filter(f, cutoff, btype="low", order=2):
    ba = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f/2.), btype=btype)
    # scipy.signal.butter(N, Wn, btype)
    # N:滤波器阶数   Wn:归一化频率（计算公式Wn=2*截止频率/采样频率）   btype:滤波器类型
    # np.divide(x1, x2): x1/x2,适用于输出值的数值类型，与输入值的数据类型无关
    # ba:输出分子和分母的系数
    # 二阶低通滤波器，采样频率为f，滤除cutoff以上的频率成分
    return DigitalFilter(ba[0], ba[1])
    # ba[0]:滤波器分子系数的矢量形式    ba[1]:滤波器分母系数的矢量形式
class DigitalFilter:
    # 创建一个类
    def __init__(self, b, a):
        # 定义构造方法，当这个类型的某个对象被创建出的时候，会自动调用这个方法（初始化方法），对对象进行初始化
        self._bs = b
        self._as = a
        self._xs = [0]*len(b)
        self._ys = [0]*(len(a)-1)
        # 创建成员变量并赋初值
    def process(self, x):
        # 定义一个process函数
        if np.isnan(x):  # ignore nans, and return as is
            # 判断x是否为NaN（未定义或不可表示的值）,是则返回x
            return x

        self._xs.insert(0, x)
        # 将变量x插入索引值为0处(即首位)
        self._xs.pop()
        # 默认移除列表中的最后一个元素
        y = (np.dot(self._bs, self._xs) / self._as[0] - np.dot(self._as[1:], self._ys))
        # np.dot:计算两个矩阵的乘积
        self._ys.insert(0, y)
        # 在self._ys首位插入变量y
        self._ys.pop()
        # 默认移除self._ys的最后一个元素
        return y

    def __call__(self, x):
        # 定义一个函数
        return self.process(x)
    # 返回值self.process()函数将x带入时的值

if __name__ == "__main__":
    # 控制函数只有直接作为脚本执行时才能执行下列过程，若调用该函数所在py文件，则不会执行下列过程
    fs = 30
    # 采样频率为30
    x = np.arange(0, 10, 1.0/fs)
    # 在0~10之间以采样频率的倒数进行抽样
    y = np.sin(2*np.pi*x) + 0.2*np.random.normal(size=len(x))
    # 生成均值为0，标准差为0，规模和x同等的高斯分布的概率密度随机数
    import pyqtgraph as pg
    # pyqtgraph：图形和用户界面库
    app = pg.QtGui.QApplication([])
    # 处理Qwidget特有的初始化和结束收尾工作
    p = pg.plot(title="test")
    # 产生一个窗口，并绘制标题为test的图
    p.plot(x, y)
    ba = scipy.signal.butter(2, 3/fs*2)
    # 二阶低通滤波器，滤除频率在3/fs*2以上的频率
    # 3/fs*2：归一化临界频率
    yfilt = scipy.signal.lfilter(ba[0], ba[1], y)
    # 使用数字滤波器过滤数据序列y
    p.plot(x, yfilt, pen=(0, 3))
    myfilt = DigitalFilter(ba[0], ba[1])
    yfilt2 = [myfilt(v) for v in y]
    # 将y中的每一个元素带入myfilt函数中进行处理？
    p.plot(x, yfilt2, pen=(1, 3))
    app.exec_()
    # 进入程序的主循环直到exit()被调用（让这个程序被运行）