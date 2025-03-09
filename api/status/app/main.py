import base64
import concurrent.futures
import json

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def show_zones():
    zones = [1, 2, 3, 6, 8, 9]

    def fetch_zone_data(zone):
        result = {}
        result["zone"] = zone
        try:
            latest_resp = requests.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca/action/latest")
            latest_resp.raise_for_status()
            result["latest"] = json.dumps(latest_resp.json())
        except Exception as e:
            result["latest"] = f"Error: {e}"
        try:
            cam1_resp = requests.get(f"http://mitacs-zone0{zone}-camera01.ccis.ualberta.ca/observation")
            cam1_resp.raise_for_status()
            result["cam1_b64"] = base64.b64encode(cam1_resp.content).decode()
        except Exception as e:
            result["cam1_b64"] = f"Error: {e}"
        try:
            cam2_resp = requests.get(f"http://mitacs-zone0{zone}-camera02.ccis.ualberta.ca/observation")
            cam2_resp.raise_for_status()
            result["cam2_b64"] = base64.b64encode(cam2_resp.content).decode()
        except Exception as e:
            result["cam2_b64"] = f"Error: {e}"
        return result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_zone_data, zones))

    html = """
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
        text-align: center;
        overflow: hidden;
      }
      body {
        font-family: sans-serif;
      }
      img {
        width: 50%;
      }
    </style>
    </head>
    <body>
      <div class="grid">
    """
    for res in results:
        cam1_src, cam1_alt = f"data:image/png;base64,{res['cam1_b64']}", "Camera1"
        if res["cam1_b64"].startswith("Error:"):
            cam1_src = "https://placehold.co/400x300"
            cam1_alt = res["cam1_b64"].replace('"', '\\"')
        cam2_src, cam2_alt = f"data:image/png;base64,{res['cam2_b64']}", "Camera2"
        if res["cam2_b64"].startswith("Error:"):
            cam2_src = "https://placehold.co/400x300"
            cam2_alt = res["cam2_b64"].replace('"', '\\"')
        html += f"""
        <div class="zone">
          <h3>Zone {res["zone"]}</h3>
          <img src="{cam1_src}" alt="{cam1_alt}"/>
          <img src="{cam2_src}" alt="{cam2_alt}"/>
          <pre>{res["latest"]}</pre>
        </div>
        """
    html += """
      </div>
    </body>
    </html>
    """
    return html
