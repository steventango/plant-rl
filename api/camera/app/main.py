import io
import time
from functools import cache
from typing import Annotated

from fastapi import Depends, FastAPI, Response

app = FastAPI()


@cache
def get_camera():
    from picamera import PiCamera
    camera = PiCamera()
    camera.resolution = (2592, 1944)
    time.sleep(2)
    return camera


@app.get("/observation", response_class=Response)
def read_observation(cam: Annotated[any, Depends(get_camera)]):
    with io.BytesIO() as buf:
        cam.capture(buf, format='png')
        image_bytes = buf.getvalue()

    headers = {"Content-Disposition": 'inline; filename="observation.png"'}
    return Response(image_bytes, headers=headers, media_type="image/png")
