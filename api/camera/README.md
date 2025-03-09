# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
rsync -azP ../install-stretch.sh zone08-camera01:~/Desktop/
rsync -azP . zone08-camera01:~/Desktop/camera
ssh zone08-camera01 -t "cd ~/Desktop && ./install-stretch.sh"
```


## Usage
```bash
# on alien
ssh zone08-camera01 -t "cd ~/Desktop/camera && docker compose up -d"
  
curl http://mitacs-zone02-camera01.ccis.ualberta.ca:8000/observation --output observation.png
```
