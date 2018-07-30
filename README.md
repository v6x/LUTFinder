# LUTFinder

You can apply LUT filter by using this module.

This module uses pyopencl, so you have to install opencl sdk first. (https://software.intel.com/en-us/articles/sdk-for-opencl-gsg)

After installing opencl sdk, follow instructions below.

1). make `PyOpenCLRunner` instance. Don't make this instance too many.
When you create This instance, You have to answer which device to use by CLI.

2). Make `LUT3DApplier` instance by using PyOpenCLRunner.

3). make `LUT3D_Extractor` instance with lut identity image. (Image has to be read by opencv2.)
The example image can be seen at `resources/sample_lut.png`. You can create identity lut image like below.
```
output = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
for r, g, b in itertools.product(range(0, 64), repeat=3):
    x = ((b % 8) * 64) + r
    y = ((b // 8) * 64) + g
    output[y, x] = [r*4, g*4 ,b*4]
Image.fromarray(output).save(f'lut_rgb.png')
```

4). Apply lut to an image by using `LUT3DApplier.execute` method and `LUT3D_Extractor` instance. 
The third parameter of `execute` means portion of applying. 
For example, if you set this parameter to 0.5, the resulting image will be average of two images, the original one and fully applied one.


Below is example code of the above. (Which can also be seen at `extract_lut3d.py`)

   
~~~
lut_img = cv2.imread('resources/sample_lut.png', cv2.IMREAD_COLOR)
lut3d = LUT3D_Extractor(lut_img).extract_lut3d()
img = cv2.imread('resources/testimage1.png', cv2.IMREAD_COLOR)
pyopencl_runner = PyOpenCLRunner()
lut_applier = LUT3DApplier(pyopencl_runner, img.nbytes)

new_img = lut_applier.execute(img, lut3d, 1.0)

cv2.imshow("gpu converted image", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

