# Lightbar Server

This starts a FastAPI server that controls the lightbar.

## Installation
```bash
# on alien
rsync -azP . pi@zone2:~/Desktop/lightbar

# on mitacs-zone2.ccis.ualberta.ca
ssh zone2
sudo apt update
sudo apt install libatlas-base-dev screen -y
cd ~/Desktop/lightbar
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage
```bash
# on alien
ssh zone2

# on mitacs-zone02-camera01.ccis.ualberta.ca
cd ~/Desktop/lightbar
source .venv/bin/activate
screen
fastapi run app/main.py --port 8000

# on alien
curl http://mitacs-zone2.ccis.ualberta.ca:8000/action -X PUT -H "Content-Type: application/json" -d '{"array": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}'
```
