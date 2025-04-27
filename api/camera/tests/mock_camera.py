import numpy as np
from PIL import Image


class MockCamera:
    def capture(self, stream, format='jpeg'):
        np.random.seed(0)
        image = np.random.rand(1944, 2592, 3) * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(stream, format=format, quality=90)
