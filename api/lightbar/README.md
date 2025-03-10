# Lightbar Server

This starts a FastAPI server that controls the lightbar.

## Installation
```bash
rsync -azP api/install-buster.sh zone8:~/Desktop/
rsync -azP api/lightbar/ zone8:~/Desktop/lightbar
ssh zone8 -t "cd ~/Desktop && ./install-buster.sh && cd lightbar && echo 'ZONE=8' > .env && docker compose up -d"
```

## Update
```bash
rsync -azP api/lightbar/ zone8:~/Desktop/lightbar
ssh zone8 -t "cd ~/Desktop/lightbar && docker compose up -d"
```


## Usage
```bash
curl http://mitacs-zone8.ccis.ualberta.ca:8080/action -X PUT -H "Content-Type: application/json" -d '{"array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}'
curl http://mitacs-zone8.ccis.ualberta.ca:8080/action/latest
```
