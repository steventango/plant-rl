import base64
import json
import os
import threading
import time
from pathlib import Path

import cv2
import litserve as ls
import numpy as np
import pytest
import requests
import supervision as sv

from ..app.main import GroundingDinoAPI

TEST_DIR = Path(__file__).parent / "test_data"


def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def start_server(port=8000):
    """Start the GroundingDinoAPI server for testing in a background thread."""
    api = GroundingDinoAPI()
    server = ls.LitServer(api, max_batch_size=2, batch_timeout=0.01)

    # Run server in a background thread
    server_thread = threading.Thread(
        target=server.run, kwargs={"port": port, "num_api_servers": 1, "generate_client_file": False}
    )
    server_thread.daemon = True
    server_thread.start()

    # Give the server a moment to start
    time.sleep(5)

    return server, server_thread


class TestGroundingDino:
    """Test class for Grounding DINO API integration tests."""

    @pytest.fixture
    def sample_image_path(self):
        return TEST_DIR / "cat.jpg"

    @pytest.fixture(scope="class")
    def api_server(self):
        """Fixture to start and stop the API server for tests."""
        port = 8932
        start_server(port=port)
        yield f"http://localhost:{port}/predict"

    def extract_detection_info(self, result):
        """Extract detection information from the API response in a consistent format."""
        boxes = result["boxes"]
        scores = result["scores"]
        text_labels = result["text_labels"]
        return boxes, scores, text_labels

    def visualize_detections(self, image_path, result):
        """
        Visualizes the detections on the image using supervision library.
        Returns the path to the saved visualization or None if no detections.
        """
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract detection data using the helper method
        boxes, confidence, text_labels = self.extract_detection_info(result)
        class_ids = list(range(len(text_labels))) if text_labels else []

        # Convert to numpy arrays
        boxes = np.array(boxes)
        confidence = np.array(confidence)
        class_ids = np.array(class_ids)

        # Create Detections object
        detections_obj = sv.Detections(
            xyxy=boxes,
            confidence=confidence,
            class_id=class_ids,
        )

        # Create annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Create labels
        labels = [f"{text_labels[i]} {confidence[i]:.2f}" for i, class_id in enumerate(class_ids)]

        # Annotate image
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections_obj)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections_obj, labels=labels)

        base_name = Path(image_path).stem
        results_dir = TEST_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{base_name}_annotated.jpg"

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), annotated_image_rgb)

    @pytest.mark.integration
    def test_predict(self, sample_image_path, api_server):
        """
        Integration test that sends a request to the Grounding DINO API.
        """

        # Send the request with a test prompt
        text_prompt = "cat"
        base64_image = encode_image(sample_image_path)
        payload = json.dumps(
            {
                "image_data": base64_image,
                "text_prompt": text_prompt,
                "threshold": 0.35,
                "text_threshold": 0.25,
            }
        )
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_server, data=payload, headers=headers)
        result = response.json()

        # Basic verification that we got a valid response
        assert result is not None, "API request failed"

        boxes, scores, text_labels = self.extract_detection_info(result)
        assert boxes == [[199.7177734375, 46.13459777832031, 499.15631103515625, 374.36907958984375]]
        assert scores == [0.3760952651500702]
        assert text_labels == ["cat"]

        self.visualize_detections(sample_image_path, result)
