# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP api/install-docker-stretch.sh zone06-camera01:~/Desktop/
ssh zone06-camera01 -t "cd ~/Desktop && ./install-docker-stretch.sh"
rsync -azP api/camera/ zone06-camera01:~/Desktop/camera
ssh zone06-camera01 -t "cd ~/Desktop/camera && docker-compose up -d"
```

## Update
```bash
rsync -azP api/camera/ zone06-camera01:~/Desktop/camera
ssh zone06-camera01 -t "cd ~/Desktop/camera && docker-compose up -d"
```


## Usage
```bash
curl http://mitacs-zone06-camera01.ccis.ualberta.ca:8080/observation --output observation.png
```
