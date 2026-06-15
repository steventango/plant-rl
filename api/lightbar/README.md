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

## On-device fallback schedule

The device can enforce a time-of-day schedule on its own so the photoperiod
still turns on/off on time when the network to the control host drops. Behaviour
is *transition-only*: each entry's calibrated action is applied once when its
`HH:MM` is crossed (in the supplied timezone) and held until the next entry, so
live `PUT /action` calls take over seamlessly during normal operation. The
schedule is persisted (`/app/schedule.json`, bind-mounted) and reloaded on
restart. The agent wrapper sets this automatically at start
(`fallback_schedule`, default on): lights on at 08:59 at 40 PPFD, off at 21:00.

```bash
# Set/replace the schedule (action is the calibrated [2,6] array, same as /action)
curl http://mitacs-zone8.ccis.ualberta.ca:8080/schedule -X PUT -H "Content-Type: application/json" \
  -d '{"timezone": "Etc/GMT-2", "entries": [
        {"time": "08:59", "action": [[0.1,0.1,0.1,0,0.1,0],[0.1,0.1,0.1,0,0.1,0]]},
        {"time": "21:00", "action": [[0,0,0,0,0,0],[0,0,0,0,0,0]]}]}'
curl http://mitacs-zone8.ccis.ualberta.ca:8080/schedule          # inspect
curl http://mitacs-zone8.ccis.ualberta.ca:8080/schedule -X DELETE # clear
```

**Redeploy required:** the schedule feature is inert until the device code is
updated. Apply the Update steps above to every active Pi.
