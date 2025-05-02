# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP api/install-docker-bookworm.sh zone02-camera02:~/Desktop/
ssh zone02-camera02 -t "cd ~/Desktop && ./install-docker-bookworm.sh"
rsync -azP api/camera2/ zone02-camera02:~/Desktop/camera2
ssh zone02-camera02 -t "cd ~/Desktop/camera2 && docker compose up -d"
```

## Update
```bash
rsync -azP api/camera2/ zone02-camera02:~/Desktop/camera2
ssh zone02-camera02 -t "cd ~/Desktop/camera2 && docker compose pull && docker compose restart"
```

## Usage
```bash
curl http://mitacs-zone02-camera02.ccis.ualberta.ca:8080/observation --output observation.jpg
```
