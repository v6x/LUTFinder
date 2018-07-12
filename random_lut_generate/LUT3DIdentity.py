import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from random_lut_generate.LUT3D import LUT3D


class LUT3DIdentity(LUT3D):
    def __init__(self):
        LUT3D.__init__(self)
        for b in range(65):
            for g in range(65):
                for r in range(65):
                    self.color_map['b'].itemset((b, g, r), min(255, b * 4))
                    self.color_map['g'].itemset((b, g, r), min(255, g * 4))
                    self.color_map['r'].itemset((b, g, r), min(255, r * 4))




