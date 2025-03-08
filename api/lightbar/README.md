# Lightbar Server

This starts a FastAPI server that controls the lightbar.

## Installation
```bash
rsync -azP . pi@zone8:~/Desktop/lightbar
ssh zone8 -t "cd ~/Desktop/lightbar && ./install.sh"
```


## Usage
```bash
ssh zone8 -t "cd ~/Desktop/lightbar && ZONE=8 docker-compose up -d"

curl http://mitacs-zone8.ccis.ualberta.ca/action -X PUT -H "Content-Type: application/json" -d '{"array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}'
```
