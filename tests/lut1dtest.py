import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from random_lut_generate.LUT1DRandomHist import LUT1DRandomHist
import cv2
import time


def lut1d_speed_test():
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


if __name__ == "__main__":
    lut1d_speed_test()