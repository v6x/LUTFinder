import abc
import matplotlib.pylab as plt
import numpy as np

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

    def apply_to_image(self, numpy_image):
        """
        :param numpy_image: a numpy array which has shape of (width, height, channels).
        channel order is BGR.
        :return:
        """
        img_shape = numpy_image.shape
        for y in range(img_shape[1]):
            for x in range(img_shape[0]):
                numpy_image[x, y, 0] = self.color_map['b'][numpy_image[x, y, 0]]
                numpy_image[x, y, 1] = self.color_map['g'][numpy_image[x, y, 1]]
                numpy_image[x, y, 2] = self.color_map['r'][numpy_image[x, y, 2]]

    def numpy_color_map(self, channel):
        return np.array(self.color_map[channel], dtype=np.uint8)


