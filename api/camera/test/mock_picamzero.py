import numpy as np


class Camera:
    def capture_array(self):
        np.random.seed(0)
        image = np.random.rand(1944, 2592, 3) * 255
        image = image.astype(np.uint8)
        return image
