# GroundingDino Server

This starts a LitServe server that serves the GroundingDino model. The server is set up to run in a Docker container. GroundingDino is a model for performing zero-shot object detection by providing text prompts.

## Installation
```bash
docker compose up -d grounding-dino
```

## Usage

### API Endpoint

```
POST http://grounding-dino:8000/predict
```

### Request Format

The API accepts a JSON payload with the following parameters:

- `image_data`: Base64 encoded image string
- `text_prompt`: Text description of the objects to detect
- `threshold`: Detection confidence threshold (optional, default value depends on model configuration)
- `text_threshold`: Text confidence threshold (optional, default value depends on model configuration)

### Example cURL Request

```bash
curl -X POST http://grounding-dino:8000/predict \
-H "Content-Type: application/json" \
-d '{"image_data": "base64_encoded_image", "text_prompt": "plant. leaf", "threshold": 0.3, "text_threshold": 0.25}'
```
