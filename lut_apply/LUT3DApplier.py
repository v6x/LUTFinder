import pyopencl as cl
import numpy as np
LUT_3D_SIZE = 65 * 65 * 65

class LUT3DApplier:
    def __init__(self, pyopencl_runner, img_size):
        # TODO Can be optimized more to use local memory but for now this is fast enough
        program_code = """
        __kernel void lut_apply(__global const uchar *img,
        __global const uchar *lut_b, __global const uchar *lut_g, __global const uchar *lut_r, 
        __global uchar *result_img)
        {
            int gid = get_global_id(0);
            uchar b = img[3 * gid];
            uchar g = img[3 * gid + 1];
            uchar r = img[3 * gid + 2];
            
            uchar b_offset = b % 4;
            uchar g_offset = g % 4;
            uchar r_offset = r % 4;
            uchar b_unoffset = 4 - b_offset;
            uchar g_unoffset = 4 - g_offset;
            uchar r_unoffset = 4 - r_offset;
            int b_idx = b / 4;
            int g_idx = g / 4;
            int r_idx = r / 4;
            
            
            float dis_1 = sqrt((float)(b_offset * b_offset + g_offset * g_offset + r_offset * r_offset));
            float dis_2 = sqrt((float)(b_offset * b_offset + g_offset * g_offset + r_unoffset * r_unoffset));
            float dis_3 = sqrt((float)(b_offset * b_offset + g_unoffset * g_unoffset + r_offset * r_offset));
            float dis_4 = sqrt((float)(b_offset * b_offset + g_unoffset * g_unoffset + r_unoffset * r_unoffset));
            float dis_5 = sqrt((float)(b_unoffset * b_unoffset + g_offset * g_offset + r_offset * r_offset));
            float dis_6 = sqrt((float)(b_unoffset * b_unoffset + g_offset * g_offset + r_unoffset * r_unoffset));
            float dis_7 = sqrt((float)(b_unoffset * b_unoffset + g_unoffset * g_unoffset + r_offset * r_offset));
            float dis_8 = sqrt((float)(b_unoffset * b_unoffset + g_unoffset * g_unoffset + r_unoffset * r_unoffset));
            
            float sum_dis = dis_1 + dis_2 + dis_3 + dis_4 + dis_5 + dis_6 + dis_7 + dis_8;
            
            float portion_1 = dis_1 / sum_dis;
            float portion_2 = dis_2 / sum_dis;
            float portion_3 = dis_3 / sum_dis;
            float portion_4 = dis_4 / sum_dis;
            float portion_5 = dis_5 / sum_dis;
            float portion_6 = dis_6 / sum_dis;
            float portion_7 = dis_7 / sum_dis;
            float portion_8 = dis_8 / sum_dis;
            
            
            int idx_1 = b_idx * 4225 + g_idx * 65 + r_idx;
            int idx_2 = b_idx * 4225 + g_idx * 65 + r_idx + 1;
            int idx_3 = b_idx * 4225 + (g_idx + 1) * 65 + r_idx;
            int idx_4 = b_idx * 4225 + (g_idx + 1) * 65 + r_idx + 1;
            int idx_5 = (b_idx + 1) * 4225 + g_idx * 65 + r_idx;
            int idx_6 = (b_idx + 1) * 4225 + g_idx * 65 + r_idx + 1;
            int idx_7 = (b_idx + 1) * 4225 + (g_idx + 1) * 65 + r_idx;
            int idx_8 = (b_idx + 1) * 4225 + (g_idx + 1) * 65 + r_idx + 1;
            
            uchar nearest_1_b = lut_b[idx_1];
            uchar nearest_2_b = lut_b[idx_2];
            uchar nearest_3_b = lut_b[idx_3];
            uchar nearest_4_b = lut_b[idx_4];
            uchar nearest_5_b = lut_b[idx_5];
            uchar nearest_6_b = lut_b[idx_6];
            uchar nearest_7_b = lut_b[idx_7];
            uchar nearest_8_b = lut_b[idx_8];
            
            uchar nearest_1_g = lut_g[idx_1];
            uchar nearest_2_g = lut_g[idx_2];
            uchar nearest_3_g = lut_g[idx_3];
            uchar nearest_4_g = lut_g[idx_4];
            uchar nearest_5_g = lut_g[idx_5];
            uchar nearest_6_g = lut_g[idx_6];
            uchar nearest_7_g = lut_g[idx_7];
            uchar nearest_8_g = lut_g[idx_8];
            
            uchar nearest_1_r = lut_r[idx_1];
            uchar nearest_2_r = lut_r[idx_2];
            uchar nearest_3_r = lut_r[idx_3];
            uchar nearest_4_r = lut_r[idx_4];
            uchar nearest_5_r = lut_r[idx_5];
            uchar nearest_6_r = lut_r[idx_6];
            uchar nearest_7_r = lut_r[idx_7];
            uchar nearest_8_r = lut_r[idx_8]; 
            
            result_img[3 * gid] = (uchar) (nearest_1_b * portion_1  
                                            + nearest_2_b * portion_2  
                                            + nearest_3_b * portion_3  
                                            + nearest_4_b * portion_4  
                                            + nearest_5_b * portion_5  
                                            + nearest_6_b * portion_6  
                                            + nearest_7_b * portion_7  
                                            + nearest_8_b * portion_8 );
            result_img[3 * gid + 1] = (uchar) (nearest_1_g * portion_1  
                                            + nearest_2_g * portion_2  
                                            + nearest_3_g * portion_3  
                                            + nearest_4_g * portion_4  
                                            + nearest_5_g * portion_5  
                                            + nearest_6_g * portion_6  
                                            + nearest_7_g * portion_7  
                                            + nearest_8_g * portion_8 );
            result_img[3 * gid + 2] = (uchar) (nearest_1_r * portion_1  
                                            + nearest_2_r * portion_2  
                                            + nearest_3_r * portion_3  
                                            + nearest_4_r * portion_4  
                                            + nearest_5_r * portion_5  
                                            + nearest_6_r * portion_6  
                                            + nearest_7_r * portion_7  
                                            + nearest_8_r * portion_8 );
        }
        """
        program = pyopencl_runner.build_program(program_code)

        self.pyopencl_runner = pyopencl_runner
        self.img_size = img_size
        self.program = program

        self.img_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, img_size)
        self.lut_b_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, LUT_3D_SIZE)
        self.lut_g_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, LUT_3D_SIZE)
        self.lut_r_buf = pyopencl_runner.alloc_buf(cl.mem_flags.READ_ONLY, LUT_3D_SIZE)
        self.result_img_buf = pyopencl_runner.alloc_buf(cl.mem_flags.WRITE_ONLY, img_size)

    def execute(self, img, lut):
        lut_b = lut.get_np_colormap('b')
        lut_g = lut.get_np_colormap('g')
        lut_r = lut.get_np_colormap('r')
        assert img.nbytes == self.img_size
        assert lut_b.nbytes == LUT_3D_SIZE
        assert lut_g.nbytes == LUT_3D_SIZE
        assert lut_r.nbytes == LUT_3D_SIZE

        pyopencl_runner = self.pyopencl_runner
        pyopencl_runner.write_buf(img, self.img_buf)
        pyopencl_runner.write_buf(lut_b, self.lut_b_buf)
        pyopencl_runner.write_buf(lut_g, self.lut_g_buf)
        pyopencl_runner.write_buf(lut_r, self.lut_r_buf)

        pixels = (img.shape[0] * img.shape[1],)
        pyopencl_runner.exec_program(self.program.lut_apply, pixels,
                                     self.img_buf, self.lut_b_buf, self.lut_g_buf, self.lut_r_buf, self.result_img_buf)
        result_img = np.empty(img.shape, dtype=img.dtype)
        pyopencl_runner.read_buf(self.result_img_buf, result_img)
        pyopencl_runner.finish()

        return result_img
