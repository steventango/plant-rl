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

        # Verify we got at least one contour
        assert len(contours) > 0, "No contours found in response"
        # Verify the first contour has points
        assert len(contours[0]) > 0, "Empty contour in response"
        # Verify we got scores
        assert len(scores) > 0, "No scores found in response"
        # Verify first score is a reasonable confidence value (between 0 and 1)
        assert 0 <= scores[0] <= 1, f"Score out of range: {scores[0]}"

        # Visualize the results
        self.visualize_masks(sample_image_path, result)
