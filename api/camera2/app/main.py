import io
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Response
from PIL import Image

app = FastAPI()


@lru_cache(maxsize=None)
def get_camera():
    from picamera2 import Picamera2
    camera = Picamera2()
    camera.configure("still")
    camera.start()
    return camera


@app.get("/observation", response_class=Response)
def read_observation(camera: Annotated[any, Depends(get_camera)]):
    array = camera.capture_array()
    image = Image.fromarray(array)

    with io.BytesIO() as buf:
        image.save(buf, "JPEG", quality=90)
        image_bytes = buf.getvalue()

    headers = {"Content-Disposition": 'inline; filename="observation.jpg"'}
    return Response(image_bytes, headers=headers, media_type="image/jpeg")
