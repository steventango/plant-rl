# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP api/install-docker-stretch.sh mitacs-zone06-camera01:~/Desktop/
ssh mitacs-zone06-camera01 -t "cd ~/Desktop && ./install-docker-stretch.sh"
rsync -azP api/camera/ mitacs-zone06-camera01:~/Desktop/camera
ssh mitacs-zone06-camera01 -t "cd ~/Desktop/camera && docker-compose up -d"
```

## Update
```bash
rsync -azP api/camera/ mitacs-zone06-camera01:~/Desktop/camera
ssh mitacs-zone06-camera01 -t "cd ~/Desktop/camera && docker-compose pull && docker-compose restart"
```


## Usage
```bash
curl http://mitacs-zone06-camera01.ccis.ualberta.ca:8080/observation --output observation.jpg
```
