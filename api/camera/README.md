# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP ../install-stretch.sh zone09-camera01:~/Desktop/
rsync -azP . zone09-camera01:~/Desktop/camera
ssh zone09-camera01 -t "cd ~/Desktop && ./install-stretch.sh && cd camera && docker-compose up -d"
```


## Usage
```bash
curl http://mitacs-zone09-camera01.ccis.ualberta.ca/observation --output observation.png
```
