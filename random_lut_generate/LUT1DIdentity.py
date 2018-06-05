from LUT1D import LUT1D


class LUT1DIdentity(LUT1D):
    def __init__(self):
        LUT1D.__init__(self)
        self.color_map['r'] = range(0, 256)
        self.color_map['g'] = range(0, 256)
        self.color_map['b'] = range(0, 256)


if __name__ == "__main__":
    lut = LUT1DIdentity()
    lut.plot()