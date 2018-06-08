import pyopencl as cl
import numpy as np
LUT_1D_SIZE = 256


class LUT1DApplier:
    def __init__(self, pyopencl_runner, img_size):
        program_code = """
        __kernel void lut_apply(__global const uchar *img,
        __global const uchar *lut_b, __global const uchar *lut_g, __global const uchar *lut_r, 
        __global uchar *result_img)
        {
            int gid = get_global_id(0);
            uchar constant_lut_b[256];
            uchar constant_lut_g[256];
            uchar constant_lut_r[256];
            if(gid < 256) {
                constant_lut_b[gid] = lut_b[gid];
                constant_lut_g[gid] = lut_g[gid];
                constant_lut_r[gid] = lut_r[gid];
            }
            result_img[3 * gid] = constant_lut_b[img[3 * gid]];
            result_img[3 * gid + 1] = constant_lut_g[img[3 * gid + 1]];
            result_img[3 * gid + 2] = constant_lut_r[img[3 * gid + 2]];
        }
        """
        program = pyopencl_runner.build_program(program_code)

        self.pyopencl_runner = pyopencl_runner
        self.img_size = img_size
        self.program = program

        self.img_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, img_size)
        self.lut_b_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, LUT_1D_SIZE)
        self.lut_g_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, LUT_1D_SIZE)
        self.lut_r_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, LUT_1D_SIZE)
        self.result_img_buf = pyopencl_runner.alloc_buf(cl.mem_flags.WRITE_ONLY, img_size)

    def execute(self, img, lut_b, lut_g, lut_r):
        assert img.nbytes == self.img_size
        assert lut_b.nbytes == LUT_1D_SIZE
        assert lut_g.nbytes == LUT_1D_SIZE
        assert lut_r.nbytes == LUT_1D_SIZE

        pyopencl_runner = self.pyopencl_runner
        pyopencl_runner.write_buf(img, self.img_buf)
        pyopencl_runner.write_buf(lut_b, self.lut_b_buf)
        pyopencl_runner.write_buf(lut_g, self.lut_g_buf)
        pyopencl_runner.write_buf(lut_r, self.lut_r_buf)

        pixel_num = img.shape[0] + img.shape[1]
        pyopencl_runner.exec_program(self.program.lut_apply, pixel_num,
                                     self.img_buf, self.lut_b_buf, self.lut_g_buf, self.lut_r_buf, self.result_img_buf)
        result_img = np.empty(img.shape, dtype=img.dtype)
        pyopencl_runner.read_buf(self.result_img_buf, result_img)

        return result_img
