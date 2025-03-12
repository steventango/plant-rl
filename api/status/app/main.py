import base64
import concurrent.futures
import html
import json
from datetime import datetime

import requests
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def show_zones():
    zones = [1, 2, 3, 6, 8, 9]

    def fetch_zone_data(zone):
        result = {}
        result["zone"] = zone
        try:
            latest_resp = requests.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca:8080/action/latest", timeout=5)
            latest_resp.raise_for_status()
            result["latest"] = json.dumps(latest_resp.json())
        except Exception as e:
            result["latest"] = f"Error: {e}"
        result["fetch_time"] = datetime.now().isoformat()
        return result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_zone_data, zones))

    response = """
    <html>
    <head>
    <style>
      .grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        margin: 0 auto;
      }
      .zone {
        border: 1px solid #ccc;
        margin: 5px;
        overflow: hidden;
      }
      h3 {
        text-align: center;
      }
      body {
        font-family: sans-serif;
      }
      img {
        width: 49%;
        aspect-ratio: 4 / 3;
        background-color: #ccc;
      }
    </style>
    </head>
    <body>
      <div class="grid">
    """
    for res in results:
        code = html.escape(res["latest"])
        fetch_time = html.escape(res["fetch_time"])
        response += f"""
        <div class="zone">
          <h3>Zone {res["zone"]}</h3>
          <img src="/proxy/zone/{res['zone']}/camera/1.png"/>
          <img src="/proxy/zone/{res['zone']}/camera/2.png"/>
          <br>
          <code>{code}</code>
          <br>
          <code>Fetched at: {fetch_time}</code>
        </div>
        """
    response += """
      </div>
    </body>
    </html>
    """
    return response


@app.get("/proxy/zone/{zone}/latest")
def proxy_latest(zone: int):
    try:
        resp = requests.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca:8080/action/latest", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/proxy/zone/{zone}/camera/{camera_id}.png")
def proxy_camera(zone: int, camera_id: int):
    resp = requests.get(
        f"http://mitacs-zone{zone:02d}-camera{camera_id:02d}.ccis.ualberta.ca:8080/observation",
        timeout=30,
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp.headers,
        media_type=resp.headers.get("Content-Type"),
    )
