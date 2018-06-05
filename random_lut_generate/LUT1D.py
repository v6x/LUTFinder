import abc
import matplotlib.pylab as plt

class LUT1D:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.color_map = dict()
        self.color_map['r'] = []
        self.color_map['g'] = []
        self.color_map['b'] = []

    def plot(self):
        xrange = range(0, 256)
        plt.plot(xrange, self.color_map['r'], 'r',
                 xrange, self.color_map['g'], 'g',
                 xrange, self.color_map['b'], 'b')
        plt.show()
