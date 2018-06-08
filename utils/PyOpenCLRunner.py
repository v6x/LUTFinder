import pyopencl as cl


class PyOpenCLRunner:
    def __init__(self):
        self.platform = None
        self.device = None
        self.context = None
        self.queue = None

    def setup(self):
        platforms = cl.get_platforms()
        print("<<<Platforms>>>")
        for i in range(len(platforms)):
            print("[%d]: " % i, platforms[i])
        platform_num = input("type platform number:")
        platform = platforms[platform_num]
        devices = platform.get_devices()
        print("<<<Devices>>>")
        for i in range(len(devices)):
            print("[%d]: " % i, devices[i])
        device_num = input("type device number:")
        device = devices[device_num]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        self.platform = platform
        self.device = device
        self.context = context
        self.queue = queue

    def build_program(self, program):
        return cl.Program(self.context, program).build()

    def alloc_buffer(self, flag, size):
        return cl.Buffer(self.context, flag, size)

    def read_buf(self, src, dst):
        cl.enqueue_read_buffer(self.queue, src, dst)

    def write_buf(self, src, dst):
        cl.enqueue_write_buffer(self.queue, src, dst)

    def exec_program(self, func, global_size, *args):
        func(self.queue, global_size, None, *args)

    def finish(self):
        self.queue.finish()