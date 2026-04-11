import numpy as np

x = np.load(r"E:\LDCT\prepared_1mm3mm_hu_-160_240\train\1mm\inputs\L067_1mm_s000_p00.npy")
y = np.load(r"E:\LDCT\prepared_1mm3mm_hu_-160_240\train\1mm\targets\L067_1mm_s000_p00.npy")

print("input:", x.shape, x.dtype, x.min(), x.max())
print("target:", y.shape, y.dtype, y.min(), y.max())