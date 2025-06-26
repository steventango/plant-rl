import base64
import json
import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import litserve as ls
import numpy as np
import pytest
import requests
import supervision as sv

sys.path.append(str(Path(__file__).parent.parent / "app"))
from main import GroundingDinoAPI

TEST_DIR = Path(__file__).parent / "test_data"


def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def run_server_proc(port=8000):
    """Run the server in a separate process."""
    api = GroundingDinoAPI()
    server = ls.LitServer(api, max_batch_size=2, batch_timeout=0.01)
    server.run(port=port, num_api_servers=1, generate_client_file=False)


def start_server(port=8000):
    """Start the GroundingDinoAPI server for testing in a background process."""
    # Run server in a background process
    server_process = multiprocessing.Process(
        target=run_server_proc,
        args=(port,),
    )
    server_process.start()
    # Give the server a moment to start
    time.sleep(5)
    return server_process


class TestGroundingDino:
    """Test class for Grounding DINO API integration tests."""

    @pytest.fixture
    def sample_image_path(self):
        return TEST_DIR / "cat.jpg"

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
        labels = [
            f"{text_labels[i]} {confidence[i]:.2f}"
            for i, class_id in enumerate(class_ids)
        ]

        # Annotate image
        annotated_image = box_annotator.annotate(
            scene=image.copy(), detections=detections_obj
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections_obj, labels=labels
        )

        base_name = Path(image_path).stem
        results_dir = TEST_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{base_name}_boxes.jpg"

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), annotated_image_rgb)

    def test_predict(self, sample_image_path):
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
        port = 8932
        server_process = start_server(port=port)
        endpoint = f"http://localhost:{port}/predict"
        response = requests.post(endpoint, data=payload, headers=headers)
        result = response.json()
        server_process.kill()

        # Basic verification that we got a valid response
        assert result is not None, "API request failed"

        boxes, scores, text_labels = self.extract_detection_info(result)
        assert np.allclose(
            boxes,
            [
                [
                    199.7177734375,
                    46.13459777832031,
                    499.15631103515625,
                    374.36907958984375,
                ]
            ],
            atol=1e-5,
        ), "Boxes do not match within tolerance"
        assert scores == pytest.approx([0.3760952651500702], rel=1e-3), (
            "Scores do not match within tolerance"
        )
        assert text_labels == ["cat"], "Text labels do not match"

        self.visualize_detections(sample_image_path, result)
