import base64
import json
import os

import cv2
import numpy as np
import requests
import supervision as sv
from PIL import Image


def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def send_request(image_path, text_prompt="cat, dog", box_threshold=0.3, text_threshold=0.25, server_url="http://localhost:8000/predict"):
    """Sends a POST request to the Grounding Dino API with the base64 encoded image and text prompt."""
    try:
        base64_image = encode_image(image_path)
        payload = json.dumps({
            "image_data": base64_image,
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(server_url, data=payload, headers=headers)

        if response.status_code == 200:
            print("Request successful!")
            result = response.json()
            print_detection_results(result, text_prompt)
            return result
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)  # Print the response text for debugging
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_detection_results(result, text_prompt):
    """Print detection results in a more readable format."""
    print(f"Detection results for prompt: '{text_prompt}'")

    # Debug the result structure
    print("Response structure:", type(result))
    print("Response content:", json.dumps(result, indent=2)[:500] + "..." if len(json.dumps(result)) > 500 else json.dumps(result, indent=2))

    # Handle direct response format (not wrapped in "detections")
    if "boxes" in result and "scores" in result and "text_labels" in result:
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        text_labels = result.get("text_labels", [])

        if len(boxes) == 0:
            print("No objects detected.")
            return

        for j in range(len(boxes)):
            box = boxes[j]
            conf = scores[j] if j < len(scores) else "N/A"
            class_name = text_labels[j] if j < len(text_labels) else "unknown"
            print(f"  {class_name} (conf: {conf:.2f}): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        return

    # Handle old format with "detections" wrapper
    detections = result.get("detections", {})

    # Handle the case where detections is a dictionary with boxes directly
    if isinstance(detections, dict) and "boxes" in detections:
        boxes = detections.get("boxes", [])
        scores = detections.get("scores", [])
        text_labels = detections.get("text_labels", [])

        if len(boxes) == 0:
            print("No objects detected.")
            return

        for j in range(len(boxes)):
            box = boxes[j]
            conf = scores[j] if j < len(scores) else "N/A"
            class_name = text_labels[j] if j < len(text_labels) else "unknown"
            print(f"  {class_name} (conf: {conf:.2f}): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    # Handle the previous expected format as fallback
    elif isinstance(detections, list):
        if not detections or len(detections) == 0:
            print("No detections found.")
            return

        # Process each detection
        for i, detection in enumerate(detections):
            print(f"\nDetection {i+1}:")

            # Handle the case where detection is a dictionary
            if isinstance(detection, dict):
                boxes = detection.get("boxes", [])
                scores = detection.get("scores", [])
                text_labels = detection.get("text_labels", [])
            else:
                # If we get here, let's print what we actually received
                print(f"  Unexpected detection format: {type(detection)}")
                print(f"  Content: {detection}")
                continue

            if len(boxes) == 0:
                print("  No objects detected.")
                continue

            for j in range(len(boxes)):
                box = boxes[j]
                conf = scores[j] if j < len(scores) else "N/A"
                class_name = text_labels[j] if j < len(text_labels) else "unknown"
                print(f"  {class_name} (conf: {conf:.2f}): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

def visualize_detections(image_path, result, output_path=None):
    """
    Visualizes the detections on the image using supervision library.

    Args:
        image_path: Path to the input image
        result: Detection results from the API (can be direct or nested in "detections")
        output_path: Path to save the output image (default: input_path_visualized.jpg)
    """
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract detection data
    boxes = []
    confidence = []
    class_ids = []
    text_labels = []

    # Handle direct response format (not wrapped in "detections")
    if "boxes" in result and "scores" in result and "text_labels" in result:
        if len(result.get("boxes", [])) > 0:
            boxes = result["boxes"]
            confidence = result["scores"]
            text_labels = result["text_labels"]
            class_ids = list(range(len(text_labels)))
    # Handle old format with "detections" wrapper
    elif "detections" in result:
        detections = result["detections"]
        # Handle the case where detections is a dictionary with boxes directly
        if isinstance(detections, dict) and "boxes" in detections:
            if len(detections.get("boxes", [])) > 0:
                boxes = detections["boxes"]
                confidence = detections["scores"]
                text_labels = detections["text_labels"]
                class_ids = list(range(len(text_labels)))
        # Handle the previous expected format as fallback
        elif isinstance(detections, list):
            for item in detections:
                if isinstance(item, dict) and len(item.get("boxes", [])) > 0:
                    boxes.extend(item["boxes"])
                    confidence.extend(item["scores"])
                    text_labels.extend(item["text_labels"])
                    class_ids.extend(list(range(len(item["text_labels"]))))

    # Create SuperVision Detections object
    if boxes:
        # Convert to xyxy format if needed
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

        # Create labels dictionary (class_id -> label text)
        labels = {
            class_id: f"{text_labels[i]} {confidence[i]:.2f}"
            for i, class_id in enumerate(class_ids)
        }

        # First annotate boxes
        annotated_image = box_annotator.annotate(
            scene=image.copy(), detections=detections_obj
        )

        # Then add labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections_obj, labels=labels
        )

        # Save or display the image
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_visualized.jpg"

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image_rgb)
        print(f"Visualization saved to {output_path}")

        return output_path
    else:
        print("No detections to visualize")
        return None

if __name__ == "__main__":
    image_path = "cat.jpg"  # Make sure cat.jpg is in the same directory, or provide full path.
    server_url = "http://localhost:8000/predict" # Or the address where your server is running

    # Example 1: Basic request with default parameters
    result = send_request(image_path,text_prompt="tail", server_url=server_url)

    # Visualize detections if we got results
    if result:
        visualize_detections(image_path, result)
