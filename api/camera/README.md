# Camera Server

This starts a FastAPI server that takes pictures when requested.

## Installation
```bash
# on alien
rsync -azP . pi@mitacs-zone02-camera01.ccis.ualberta.ca:~/Desktop/camera

# on mitacs-zone02-camera01.ccis.ualberta.ca
ssh pi@mitacs-zone02-camera01.ccis.ualberta.ca
sudo apt update
sudo apt install python3-picamzero screen -y
cd ~/Desktop/camera
python3 -m venv .venv
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage
```bash
# on alien
ssh pi@mitacs-zone02-camera01.ccis.ualberta.ca

# on mitacs-zone02-camera01.ccis.ualberta.ca
cd ~/Desktop/camera
source .venv/bin/activate
export PYTHONIOENCODING=utf-8
screen
fastapi run app/main.py --port 8000

# on alien
curl http://mitacs-zone02-camera01.ccis.ualberta.ca:8000/observation --output observation.png
```
