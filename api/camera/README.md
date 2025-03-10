# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP api/install-stretch.sh zone09-camera01:~/Desktop/
rsync -azP pi/camera zone08-camera01:~/Desktop/camera
ssh zone09-camera01 -t "cd ~/Desktop && ./install-stretch.sh && cd camera && docker-compose up -d"
```

## Update
```bash
rsync -azP api/camera zone08-camera01:~/Desktop/camera
ssh zone08-camera01 -t "cd ~/Desktop/camera && docker-compose up -d"
```


## Usage
```bash
curl http://mitacs-zone09-camera01.ccis.ualberta.ca:8080/observation --output observation.png
```
