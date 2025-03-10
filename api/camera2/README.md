# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP api/install-stretch.sh zone02-camera01:~/Desktop/
rsync -azP api/camera2 zone02-camera01:~/Desktop/camera2
ssh zone02-camera01 -t "cd ~/Desktop && ./install-stretch.sh && cd camera2 && docker-compose up -d"
```

## Update
```bash
rsync -azP api/camera2 zone02-camera01:~/Desktop/camera2
ssh zone02-camera01 -t "cd ~/Desktop/camera2 && docker compose up -d"
```

## Usage
```bash
curl http://mitacs-zone02-camera01.ccis.ualberta.ca:8080/observation --output observation.png
```
