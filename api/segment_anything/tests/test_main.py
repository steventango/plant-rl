import base64
import json
import multiprocessing
import time
from pathlib import Path

import cv2
import litserve as ls
import numpy as np
import pytest
import requests
import supervision as sv
from supervision.annotators.utils import ColorLookup

from ..app.main import SegmentAnythingAPI

TEST_DIR = Path(__file__).parent / "test_data"


def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def run_server_proc(port=8000):
    """Run the server in a separate process."""
    api = SegmentAnythingAPI()
    server = ls.LitServer(api, max_batch_size=2, batch_timeout=0.01)
    server.run(port=port, num_api_servers=1, generate_client_file=False)


def start_server(port=8000):
    """Start the SegmentAnythingAPI server for testing in a background process."""
    # Run server in a background process
    server_process = multiprocessing.Process(target=run_server_proc, args=(port,))
    server_process.start()

    # Give the server a moment to start
    time.sleep(5)

    return server_process


class TestSegmentAnything:
    """Test class for Segment Anything API integration tests."""

    @pytest.fixture
    def sample_image_path(self):
        return TEST_DIR / "cat.jpg"

    def extract_mask_info(self, result):
        """Extract mask information from the API response in a consistent format."""
        contours = result["contours"]
        scores = result["scores"]
        return contours, scores

    def visualize_masks(self, image_path, result):
        """
        Visualizes the masks on the image using supervision library.
        Returns the path to the saved visualization or None if no masks.
        """
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract contours and scores
        contours, scores = self.extract_mask_info(result)

        if not contours:
            return None

        # Create a blank canvas for mask visualization
        mask_canvas = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Draw masks on canvas
        for i, contour in enumerate(contours):
            if contour:
                # Convert contour to numpy array with correct shape for fillPoly
                contour_np = np.array(contour, dtype=np.int32)
                cv2.fillPoly(mask_canvas, [contour_np], i + 1)

        # Use Supervision library for visualization
        mask_annotator = sv.MaskAnnotator(color_lookup=ColorLookup.INDEX)

        # Create detections object with the masks
        detections = sv.Detections(
            xyxy=np.array([[0, 0, image.shape[1], image.shape[0]]]),  # dummy bounding box
            mask=np.array([mask_canvas > 0]),
        )

        # Annotate image with masks
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

        # Save the result
        base_name = Path(image_path).stem
        results_dir = TEST_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{base_name}_masks.jpg"

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), annotated_image_rgb)

        return output_path

    def test_predict(self, sample_image_path):
        """
        Integration test that sends a request to the Segment Anything API.
        """
        # Define a bounding box for the cat in the image
        # Format is [x1, y1, x2, y2]
        boxes = [[200, 50, 500, 375]]  # Approximate box around the cat

        # Send the request with a test box
        base64_image = encode_image(sample_image_path)
        payload = json.dumps(
            {
                "image_data": base64_image,
                "boxes": boxes,
                "multimask_output": False,
            }
        )
        headers = {"Content-Type": "application/json"}
        port = 8933  # Different port from Grounding DINO test
        server_process = start_server(port=port)
        endpoint = f"http://localhost:{port}/predict"
        response = requests.post(endpoint, data=payload, headers=headers)
        server_process.kill()
        result = response.json()

        # Basic verification that we got a valid response
        assert result is not None, "API request failed"
        assert "contours" in result, "Response doesn't contain contours"
        assert "scores" in result, "Response doesn't contain scores"

        contours, scores = self.extract_mask_info(result)

        assert contours == [
            [
                [430, 47],
                [429, 48],
                [427, 48],
                [426, 49],
                [426, 50],
                [425, 51],
                [425, 53],
                [424, 54],
                [424, 58],
                [423, 59],
                [423, 64],
                [422, 65],
                [422, 67],
                [423, 68],
                [423, 74],
                [424, 75],
                [424, 78],
                [425, 79],
                [425, 80],
                [426, 81],
                [427, 81],
                [430, 84],
                [430, 85],
                [437, 92],
                [438, 92],
                [439, 93],
                [440, 93],
                [446, 99],
                [447, 99],
                [448, 100],
                [448, 101],
                [449, 102],
                [450, 102],
                [455, 107],
                [455, 108],
                [459, 112],
                [459, 113],
                [460, 114],
                [460, 115],
                [461, 116],
                [461, 120],
                [460, 121],
                [459, 121],
                [456, 124],
                [455, 124],
                [454, 125],
                [453, 125],
                [452, 126],
                [450, 126],
                [449, 127],
                [448, 127],
                [446, 129],
                [444, 129],
                [443, 130],
                [442, 130],
                [441, 131],
                [440, 131],
                [439, 132],
                [438, 132],
                [436, 134],
                [435, 134],
                [432, 137],
                [431, 137],
                [430, 138],
                [429, 138],
                [428, 139],
                [427, 139],
                [425, 141],
                [424, 141],
                [422, 143],
                [420, 143],
                [419, 144],
                [418, 144],
                [417, 145],
                [416, 145],
                [415, 146],
                [413, 146],
                [412, 147],
                [411, 147],
                [410, 148],
                [408, 148],
                [407, 149],
                [405, 149],
                [403, 151],
                [402, 151],
                [399, 154],
                [398, 154],
                [397, 155],
                [394, 155],
                [393, 154],
                [392, 154],
                [391, 155],
                [390, 154],
                [390, 148],
                [389, 148],
                [388, 149],
                [387, 149],
                [384, 152],
                [383, 152],
                [382, 153],
                [381, 153],
                [380, 154],
                [378, 154],
                [377, 155],
                [375, 155],
                [374, 156],
                [373, 156],
                [372, 155],
                [364, 155],
                [363, 154],
                [358, 154],
                [357, 155],
                [355, 155],
                [354, 156],
                [353, 156],
                [353, 157],
                [352, 158],
                [349, 155],
                [349, 154],
                [343, 148],
                [342, 148],
                [341, 149],
                [338, 149],
                [336, 151],
                [335, 151],
                [333, 153],
                [333, 154],
                [332, 155],
                [332, 158],
                [331, 159],
                [330, 159],
                [328, 161],
                [327, 161],
                [326, 162],
                [325, 162],
                [322, 165],
                [321, 165],
                [321, 166],
                [318, 169],
                [310, 169],
                [309, 170],
                [307, 170],
                [306, 171],
                [295, 171],
                [294, 172],
                [287, 172],
                [286, 173],
                [284, 173],
                [283, 174],
                [276, 174],
                [275, 175],
                [271, 175],
                [270, 176],
                [268, 176],
                [267, 177],
                [265, 177],
                [264, 178],
                [257, 178],
                [256, 177],
                [255, 177],
                [254, 176],
                [252, 176],
                [251, 175],
                [246, 175],
                [245, 174],
                [243, 174],
                [242, 173],
                [241, 173],
                [240, 172],
                [239, 172],
                [238, 171],
                [236, 171],
                [235, 170],
                [234, 170],
                [233, 169],
                [232, 169],
                [231, 168],
                [230, 168],
                [229, 167],
                [228, 167],
                [227, 166],
                [226, 166],
                [225, 165],
                [224, 165],
                [223, 164],
                [222, 164],
                [221, 163],
                [220, 163],
                [219, 162],
                [218, 162],
                [216, 160],
                [215, 160],
                [211, 156],
                [210, 156],
                [210, 158],
                [211, 159],
                [211, 161],
                [212, 162],
                [212, 164],
                [213, 165],
                [213, 167],
                [214, 168],
                [214, 169],
                [215, 170],
                [215, 171],
                [216, 172],
                [216, 173],
                [218, 175],
                [218, 176],
                [219, 177],
                [219, 178],
                [220, 179],
                [220, 181],
                [221, 182],
                [221, 184],
                [222, 185],
                [222, 187],
                [223, 188],
                [223, 189],
                [224, 190],
                [224, 191],
                [225, 192],
                [225, 193],
                [226, 194],
                [226, 203],
                [225, 204],
                [225, 210],
                [226, 211],
                [226, 219],
                [225, 220],
                [225, 223],
                [224, 224],
                [224, 228],
                [223, 229],
                [223, 230],
                [222, 231],
                [222, 232],
                [221, 233],
                [221, 235],
                [220, 236],
                [220, 240],
                [219, 241],
                [219, 245],
                [220, 246],
                [219, 247],
                [219, 265],
                [221, 267],
                [221, 268],
                [222, 269],
                [222, 270],
                [223, 271],
                [223, 272],
                [224, 273],
                [224, 274],
                [225, 275],
                [225, 276],
                [226, 277],
                [226, 278],
                [232, 284],
                [232, 285],
                [235, 288],
                [235, 289],
                [236, 289],
                [237, 290],
                [237, 291],
                [240, 294],
                [240, 295],
                [242, 297],
                [242, 298],
                [244, 300],
                [244, 301],
                [245, 302],
                [245, 308],
                [246, 309],
                [246, 314],
                [247, 315],
                [246, 316],
                [246, 318],
                [247, 319],
                [247, 320],
                [248, 321],
                [248, 324],
                [253, 329],
                [253, 330],
                [256, 333],
                [257, 333],
                [258, 334],
                [259, 334],
                [260, 335],
                [261, 335],
                [262, 336],
                [263, 336],
                [264, 337],
                [265, 337],
                [266, 338],
                [266, 339],
                [267, 339],
                [269, 341],
                [268, 342],
                [268, 347],
                [271, 350],
                [272, 350],
                [274, 352],
                [274, 353],
                [275, 354],
                [275, 359],
                [276, 360],
                [276, 361],
                [278, 363],
                [278, 364],
                [279, 365],
                [280, 365],
                [281, 366],
                [282, 366],
                [283, 367],
                [284, 367],
                [285, 368],
                [293, 368],
                [294, 369],
                [296, 369],
                [297, 370],
                [298, 370],
                [299, 371],
                [301, 371],
                [302, 372],
                [304, 372],
                [306, 374],
                [419, 374],
                [423, 370],
                [423, 369],
                [424, 368],
                [424, 366],
                [425, 365],
                [425, 364],
                [426, 363],
                [426, 362],
                [427, 361],
                [427, 360],
                [428, 359],
                [428, 357],
                [429, 356],
                [429, 354],
                [430, 353],
                [430, 352],
                [434, 348],
                [434, 347],
                [437, 344],
                [437, 343],
                [438, 342],
                [439, 342],
                [442, 339],
                [443, 339],
                [444, 338],
                [444, 337],
                [445, 336],
                [445, 335],
                [446, 334],
                [447, 334],
                [447, 333],
                [451, 329],
                [452, 329],
                [455, 326],
                [456, 326],
                [457, 325],
                [459, 325],
                [462, 322],
                [464, 322],
                [464, 321],
                [465, 320],
                [466, 320],
                [467, 321],
                [471, 317],
                [473, 319],
                [475, 319],
                [477, 317],
                [478, 317],
                [476, 315],
                [476, 314],
                [477, 313],
                [482, 313],
                [483, 312],
                [484, 313],
                [484, 314],
                [485, 315],
                [486, 314],
                [489, 314],
                [490, 315],
                [491, 315],
                [492, 314],
                [493, 314],
                [494, 315],
                [494, 316],
                [493, 317],
                [493, 318],
                [492, 319],
                [494, 321],
                [493, 322],
                [493, 338],
                [494, 339],
                [494, 340],
                [495, 341],
                [498, 341],
                [499, 342],
                [499, 127],
                [498, 126],
                [498, 125],
                [497, 124],
                [497, 123],
                [496, 122],
                [496, 121],
                [494, 119],
                [494, 118],
                [492, 116],
                [492, 114],
                [488, 110],
                [488, 109],
                [487, 108],
                [487, 107],
                [486, 106],
                [486, 105],
                [485, 104],
                [485, 103],
                [482, 100],
                [481, 100],
                [480, 99],
                [480, 98],
                [479, 97],
                [479, 96],
                [476, 93],
                [476, 92],
                [474, 90],
                [473, 90],
                [471, 88],
                [470, 88],
                [469, 87],
                [469, 86],
                [468, 85],
                [468, 84],
                [467, 83],
                [466, 83],
                [465, 82],
                [461, 82],
                [460, 81],
                [460, 80],
                [458, 78],
                [457, 78],
                [453, 74],
                [453, 73],
                [452, 73],
                [451, 72],
                [450, 72],
                [444, 66],
                [442, 66],
                [441, 65],
                [441, 64],
                [440, 63],
                [440, 62],
                [439, 61],
                [439, 60],
                [437, 58],
                [437, 56],
                [436, 55],
                [436, 53],
                [435, 52],
                [435, 50],
                [434, 50],
                [432, 48],
                [431, 48],
            ]
        ]
        assert scores == [0.9227880835533142]

        # Visualize the results
        self.visualize_masks(sample_image_path, result)
