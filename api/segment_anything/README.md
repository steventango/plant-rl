# Segment Anything Model (SAM2) Server

This starts a LitServe server that serves the Segment Anything Model (SAM2). The server is set up to run in a Docker container. SAM2 is a foundation model for image segmentation that can generate high-quality object masks from input bounding boxes.

## Installation
```bash
docker compose up -d segment-anything
```

## Usage

### API Endpoint

```
POST http://segment-anything:8000/predict
```

### Request Format

The API accepts a JSON payload with the following parameters:

- `image_data`: Base64 encoded image string
- `boxes`: List of bounding boxes in the format [[x1, y1, x2, y2], ...] where each box defines the region to segment
- `multimask_output`: Boolean flag to indicate whether to return multiple mask predictions per box (optional, default: false)

### Example cURL Request

```bash
curl -X POST http://segment-anything:8000/predict \
-H "Content-Type: application/json" \
-d '{"image_data": "base64_encoded_image", "boxes": [[0, 0, 100, 100]], "multimask_output": false}'
```

### Response Format

The API returns a JSON object with the following fields:

- `contours`: List of contours for the segmented objects. Each contour is a list of [x,y] points representing the boundary of the object.
- `scores`: List of confidence scores for each contour

This API uses contours instead of full masks to significantly reduce bandwidth usage. The contour representation provides significant bandwidth savings compared to transmitting full binary masks, especially for large images.
