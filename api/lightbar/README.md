# Lightbar Server

This starts a FastAPI server that controls the lightbar.

## Installation
```bash
rsync -azP api/install-docker-buster.sh mitacs-zone8:~/Desktop/
ssh mitacs-zone8 -t "cd ~/Desktop && ./install-docker-buster.sh"
rsync -azP api/lightbar/ mitacs-zone8:~/Desktop/lightbar
ssh mitacs-zone8 -t "cd ~/Desktop/lightbar && echo 'ZONE=mitacs-zone8' > .env && docker compose up -d"
```

## Update
```bash
rsync -azP api/lightbar/ mitacs-zone8:~/Desktop/lightbar
ssh mitacs-zone8 -t "cd ~/Desktop/lightbar && docker compose pull && docker compose restart"
```


## Usage
```bash
curl http://mitacs-zone8.ccis.ualberta.ca:8080/action -X PUT -H "Content-Type: application/json" -d '{"array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}'
curl http://mitacs-zone8.ccis.ualberta.ca:8080/action/latest
```
