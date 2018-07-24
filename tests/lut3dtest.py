import os
import sys
import cv2
import time
import unittest
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from random_lut_generate.LUT3DIdentity import LUT3DIdentity
from lut_apply.LUT3DApplier import LUT3DApplier
from utils.PyOpenCLRunner import PyOpenCLRunner


class LUT3DApplyTest(unittest.TestCase):

    def test_lut3d_identity(self):
        print("testing lut 3d identity")
        lut3d = LUT3DIdentity()
        img = cv2.imread('../resources/testimage1.png', cv2.IMREAD_COLOR)

        pyopencl_runner = PyOpenCLRunner()
        lut_applier = LUT3DApplier(pyopencl_runner, img.nbytes)

        test_num = 10
        start_time = time.time()
        for i in range(test_num):
            copied_img = img[:]
            new_img = lut_applier.execute(copied_img, lut3d, 1, 1.0)

        elapsed_time = time.time() - start_time
        # currently about 0.005 sec per processing one 1280x720 image
        print("iter %d, total %f sec elapsed, %f sec per iter" % (test_num, elapsed_time, elapsed_time / test_num))

        cv2.imshow("gpu converted image", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    unittest.main()