import io
from typing import Annotated

from fastapi import Depends, FastAPI, Response
from PIL import Image

app = FastAPI()


def get_camera():
    from picamzero import Camera

    return Camera()


@app.get("/observation", response_class=Response)
def read_observation(cam: Annotated[any, Depends(get_camera)]):
    array = cam.capture_array()
    image = Image.fromarray(array)

    with io.BytesIO() as buf:
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

    headers = {"Content-Disposition": 'inline; filename="observation.jpg"'}
    return Response(image_bytes, headers=headers, media_type="image/jpg")
