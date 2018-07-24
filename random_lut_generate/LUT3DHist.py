import os
import sys
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.UtilFunctions import interpolate_from_ndarray
from random_lut_generate.LUT3D import LUT3D


def fill_hists(hists, filled, coord):
    x, y, z = coord
    if x < 0 or y < 0 or z < 0:
        return 0
    if filled.item(x, y, z) is not 0:
        return hists.item((coord))
    sum = hists.item(coord)
    sum += fill_hists((x - 1, y, z)) + fill_hists((x, y - 1, z)) + fill_hists((x, y, z - 1))
    hists.itemset(coord, sum)
    filled.itemset(coord, 1)
    return sum

class LUT3DRandomHist(LUT3D):
    def __init__(self, hist_dis):
        LUT3D.__init__(self)
        for channel in ['r', 'g', 'b']:
            self._init_color_map(channel, hist_dis)

    def _init_color_map(self, channel, hist_dis):
        hists = np.ndarray(shape=(64, 64, 64), dtype=np.float)
        for x in range(0, 64, hist_dis):
            for y in range(0, 64, hist_dis):
                for z in range(0, 64, hist_dis):
                    hists.itemset((x, y, z), random.uniform(0, 1.0))

        for x in range(64):
            for y in range(64):
                for z in range(64):
                    hists.itemset((x, y, z), interpolate_from_ndarray(hists, (x, y, z), hist_dis))

        filled = np.zeros((64, 64, 64))
        fill_hists(hists, filled, (63, 63, 63))
        normalize_factor = 255 / hists.item((63, 63, 63))
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    prev_val = hists.item((x, y, z))
                    norm_x = x / 63
                    norm_y = y / 63
                    norm_z = z / 63
                    hists.itemset((x, y, z), normalize_factor * prev_val * (norm_x + norm_y + norm_z) / (3 * norm_x * norm_y * norm_z))

        self.color_map[channel] = hists.astype(np.uint8)

if __name__ == "__main__":
    lut = LUT3DRandomHist(4)
