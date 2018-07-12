import sys
import cv2
import itertools
from random_lut_generate.LUT3D import LUT3D
from utils.PyOpenCLRunner import PyOpenCLRunner
from lut_apply.LUT3DApplier import LUT3DApplier

class LUT3D_Extractor:
    def __init__(self, img):
        self.img = img

    def extract_lut3d(self):
        lut3d = LUT3D()
        for r, g, b in itertools.product(range(0, 65), repeat=3):
            r_r = min(r, 63)
            r_g = min(g, 63)
            r_b = min(b, 63)
            x = ((r_b % 8) * 64) + r_r
            y = ((r_b // 8) * 64) + r_g
            lut3d.color_map['b'][b, g, r] = self.img[y, x, 0]
            lut3d.color_map['g'][b, g, r] = self.img[y, x, 1]
            lut3d.color_map['r'][b, g, r] = self.img[y, x, 2]
        return lut3d


if __name__ == "__main__":
    img_path = sys.argv[1]
    lut_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    lut3d = LUT3D_Extractor(lut_img).extract_lut3d()
    img = cv2.imread('resources/testimage1.png', cv2.IMREAD_COLOR)

    pyopencl_runner = PyOpenCLRunner()
    lut_applier = LUT3DApplier(pyopencl_runner, img.nbytes)

    new_img = lut_applier.execute(img, lut3d)

    cv2.imshow("gpu converted image", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
