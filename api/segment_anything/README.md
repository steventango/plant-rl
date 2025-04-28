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

- `masks`: List of binary masks for the segmented objects
- `scores`: List of confidence scores for each mask
- `logits`: List of logit values for each mask

Each mask is a 2D binary array where 1 indicates the object pixels and 0 indicates the background.
