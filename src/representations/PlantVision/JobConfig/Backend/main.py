from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import cv2
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import numpy as np
import base64
from PIL import Image
import io


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class undistortFisheyeData(BaseModel):
    img: str

    fx: float
    fy: float
    cx: float
    cy: float

    k1: float
    k2: float
    k3: float
    k4: float

class undistortRectilinearData(BaseModel):
    img: str

    fx: float
    fy: float
    cx: float
    cy: float

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

class histogramData(BaseModel):
    img: str

class thresholdData(BaseModel):
    img: str

    hl: float
    hh: float
    sl: float
    sh: float
    vl: float
    vh: float

    fill: float

    colored: bool
    invert: bool


@app.get("/")
def read_root():
    return {"Hello": "World"}


def decodeb64(b64str: str):
    # returns an np array image
    b64str = b64str[len("data:image/png;base64,"):]
    # with open('readme.txt', 'w') as f:
    #     f.write(b64str)
    
    base64_decoded = base64.b64decode(b64str)
    image = Image.open(io.BytesIO(base64_decoded))
    return np.array(image)

def encodeb64(arr: np.ndarray):
    # returns a b64str
    image = Image.fromarray(arr.astype("uint8"))
    rawBytes = io.BytesIO()
    image.save(rawBytes, "PNG")
    rawBytes.seek(0)
    return base64.b64encode(rawBytes.read())

@app.post("/undistort-rectilinear")
def undistort(data: undistortRectilinearData):
    K = np.array([
        [data.fx, 0.0, data.cx],
        [0.0, data.fy, data.cy],
        [0.0, 0.0, 1.0],
    ])
    D = np.array([data.k1, data.k2, data.p1, data.p2, data.k3])

    # print(K, D)
    image = decodeb64(data.img)

    distorted_img = image

    balance = 1
    dim = (distorted_img.shape[1], distorted_img.shape[0])

    # rectilinear
    new_K, validPixRoi = cv2.getOptimalNewCameraMatrix(K, D, dim, balance)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_32FC1)

    # and then remap:
    undistorted = cv2.remap(distorted_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return encodeb64(undistorted)


@app.post("/undistort-fisheye")
def undistort(data: undistortFisheyeData):
    K = np.array([
        [data.fx, 0.0, data.cx],
        [0.0, data.fy, data.cy],
        [0.0, 0.0, 1.0],
    ])
    D = np.array([data.k1, data.k2, data.k3, data.k4])

    # print(K, D)
    image = decodeb64(data.img)

    distorted_img = image

    balance = 1
    dim = (distorted_img.shape[1], distorted_img.shape[0])

    # fisheye
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_32FC1)


    # and then remap:
    undistorted = cv2.remap(distorted_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return encodeb64(undistorted)

@app.post("/threshold")
def threshold(data: thresholdData):
    image = decodeb64(data.img)

    # Returns a binary image with green pixels as white, and the rest black.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    upper = np.array([data.hh, data.sh, data.vh])
    lower = np.array([data.hl, data.sl, data.vl])
    
    mask = cv2.inRange(hsv, lower, upper)

    if data.invert:
        mask = cv2.bitwise_not(mask, mask)

    # should invert be before or after fill???
    try:
        mask = pcv.fill(mask, data.fill)
    except:
        pass


    if data.colored:
        band = cv2.bitwise_and(hsv, hsv, mask=mask)
        converted = cv2.cvtColor(band, cv2. COLOR_HSV2RGB)
        return encodeb64(converted)
    else:
        return encodeb64(mask)
    
@app.get("/ping")
def ping():
    return "Success"


def create_histogram(data: histogramData):
    img = decodeb64(data.img)

    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
    # hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    # hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    plt.plot(hist_h, color='r', label="h")
    # plt.plot(hist_s, color='g', label="s")
    # plt.plot(hist_v, color='b', label="v")
    # plt.legend()
    # plt.show()
    plt.title("Hue histogram")

    imgdata = io.BytesIO()
    plt.savefig(imgdata, format='png')
    plt.close()

    return imgdata
    
@app.post('/histogram')
async def get_img(data: histogramData, background_tasks: BackgroundTasks):
    img_buf = create_histogram(data)
    # get the entire buffer content
    # because of the async, this will await the loading of all content
    img_buf.seek(0)
    res = base64.b64encode(img_buf.read())
    background_tasks.add_task(img_buf.close)
    return res

