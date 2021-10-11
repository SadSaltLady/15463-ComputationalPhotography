import numpy as np
from skimage import io
grad = np.tile(np.linspace(0, 1, 255), (255,1))
io.imsave("what.jpg", grad)