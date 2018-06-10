import os
import sys
import cv2
import time
import unittest
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from random_lut_generate.LUT1DRandomHist import LUT1DRandomHist
from lut_apply.LUT1DApplier import LUT1DApplier
from utils.PyOpenCLRunner import PyOpenCLRunner


class LUT1DApplyTest(unittest.TestCase):

    def test_lut1d_cpu_speed_test(self):
        print("testing lut 1d cpu speed")
        lut1d = LUT1DRandomHist(5)
        img = cv2.imread('../resources/testimage1.png', cv2.IMREAD_COLOR)
        test_num = 10

        start_time = time.time()
        for i in range(test_num):
            copied_img = img[:]
            lut1d.apply_to_image(copied_img)
        elapsed_time = time.time() - start_time
        # currently about 1 sec per processing one 1280x720 image
        print("iter %d, total %f sec elapsed, %f sec per iter" % (test_num, elapsed_time, elapsed_time / test_num))

    def test_lut1d_gpu_speed_test(self):
        print("testing lut 1d gpu speed")
        lut1d = LUT1DRandomHist(5)
        img = cv2.imread('../resources/testimage1.png', cv2.IMREAD_COLOR)
        test_num = 10

        pyopencl_runner = PyOpenCLRunner()
        lut_applier = LUT1DApplier(pyopencl_runner, img.nbytes)
        start_time = time.time()
        for i in range(test_num):
            copied_img = img[:]
            lut_applier.execute(copied_img, lut1d)

        elapsed_time = time.time() - start_time
        # currently about 0.005 sec per processing one 1280x720 image
        print("iter %d, total %f sec elapsed, %f sec per iter" % (test_num, elapsed_time, elapsed_time / test_num))

    def test_lut1d_cpu_gpu_compare(self):
        print("testing comparasion of lut 1d gpu and cpu result")
        lut1d = LUT1DRandomHist(5)
        img = cv2.imread('../resources/testimage1.png', cv2.IMREAD_COLOR)

        pyopencl_runner = PyOpenCLRunner()
        lut_applier = LUT1DApplier(pyopencl_runner, img.nbytes)
        new_img = lut_applier.execute(img, lut1d)
        cv2.imshow("gpu converted image", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        lut1d.apply_to_image(img)
        cv2.imshow("cpu converted image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()