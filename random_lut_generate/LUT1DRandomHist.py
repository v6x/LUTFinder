import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.QuadraticFunction import QuadraticFunction
from utils.LinearFunction import get_linearfunction_from_two_point
from LUT1D import LUT1D
import random

def random_0_to_1():
    val = random.uniform(0., 1.)
    # below loop makes curve more dynamically
    while(0.2 < val and  val < 0.8):
        val = random.uniform(0., 1.)
    return val

def make_random_hists(hist_num):
    hists = []
    for i in range(hist_num):
        random_float = random_0_to_1()
        point_pair = (1. / (hist_num - 1) * i, random_float)
        hists.append(point_pair)
    return hists


def make_linear_functions_from_hists(hists):
    linear_funcs = []
    for i in range(len(hists) - 1):
        linear_func = get_linearfunction_from_two_point(hists[i], hists[i + 1])
        x_range = (hists[i][0], hists[i+1][0])
        linear_funcs.append((x_range, linear_func))
    return linear_funcs


def integrate_linear_funcs(linear_funcs):
    integrated_funcs = []
    start_y = 0
    for x_range, linear_func in linear_funcs:
        integrated = linear_func.get_integrate()
        start_point = (x_range[0], start_y)
        integrated.vertical_move(start_point)
        integrated_funcs.append((x_range, integrated))

        start_y = integrated.calc(x_range[1])
    end_y = start_y
    return integrated_funcs, end_y


class LUT1DRandomHist(LUT1D):
    def __init__(self, hist_num):
        LUT1D.__init__(self)
        for channel in ['r', 'g', 'b']:
            self._init_color_map(channel, hist_num)

    def _init_color_map(self, channel, hist_num):
        hists = make_random_hists(hist_num)
        linear_funcs = make_linear_functions_from_hists(hists)
        integrated_funcs, end_y = integrate_linear_funcs(linear_funcs)
        for x_range, integrated_func in integrated_funcs:
            denormalized_start_x = int(x_range[0] * 256)
            denormalized_end_x = int(x_range[1] * 256)
            for x in range(denormalized_start_x, denormalized_end_x):
                y = integrated_func.calc(x / 256.)
                normalized_y = y * 256. / end_y
                self.color_map[channel].append(normalized_y)
        assert len(self.color_map[channel]) == 256


if __name__ == "__main__":
    lut = LUT1DRandomHist(5)
    lut.plot()
