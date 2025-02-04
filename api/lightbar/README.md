# Lightbar Server

This starts a FastAPI server that controls the lightbar.

## Installation
```bash
# on alien
rsync -azP . pi@zone2:~/Desktop/lightbar

# on mitacs-zone2.ccis.ualberta.ca
ssh zone2
sudo apt update
sudo apt install libc6 libopenblas-dev screen -y
cd ~/Desktop/lightbar
python3.11 -m venv .venv
sed -i 's/include-system-site-packages = true/include-system-site-packages = false/' .venv/pyvenv.cfg
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
curl http://mitacs-zone02-camera01.ccis.ualberta.ca:8000/action -X POST -H "Content-Type: application/json" -d '{"action": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}'
```
