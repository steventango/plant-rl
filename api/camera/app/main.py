import io
from typing import Annotated

from fastapi import Depends, FastAPI, Response
from PIL import Image

app = FastAPI()
camera = None

def get_camera():
    from picamera2 import Picamera2
    global camera
    if camera is not None:
        return camera
    camera = Picamera2()
    camera.configure("still")
    camera.start()
    return camera


@app.get("/observation", response_class=Response)
def read_observation(cam: Annotated[any, Depends(get_camera)]):
    array = cam.capture_array()
    image = Image.fromarray(array)

    with io.BytesIO() as buf:
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

    headers = {"Content-Disposition": 'inline; filename="observation.jpg"'}
    return Response(image_bytes, headers=headers, media_type="image/jpg")
