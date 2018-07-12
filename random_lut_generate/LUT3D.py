import abc
import matplotlib.pylab as plt
import numpy as np

class LUT3D:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.color_map = dict()
        self.color_map['r'] = np.ndarray(shape=(65, 65, 65), dtype=np.uint8)
        self.color_map['g'] = np.ndarray(shape=(65, 65, 65), dtype=np.uint8)
        self.color_map['b'] = np.ndarray(shape=(65, 65, 65), dtype=np.uint8)

    def get_np_colormap(self, channel):
        return self.color_map[channel]


