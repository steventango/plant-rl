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
        result["latest"] = json.dumps(requests.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca/action/latest").json(), indent=4)
        cam1_resp = requests.get(f"http://mitacs-zone0{zone}-camera01.ccis.ualberta.ca/observation")
        cam2_resp = requests.get(f"http://mitacs-zone0{zone}-camera02.ccis.ualberta.ca/observation")
        result["cam1_b64"] = base64.b64encode(cam1_resp.content).decode()
        result["cam2_b64"] = base64.b64encode(cam2_resp.content).decode()
        return result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_zone_data, zones))

    html = """
    <html>
    <head>
    <style>
      .grid { display: grid; grid-template-columns: repeat(3, 1fr); }
      .zone { border: 1px solid #ccc; margin: 5px; text-align: center; }
    </style>
    </head>
    <body>
      <div class="grid">
    """
    for res in results:
        html += f"""
        <div class="zone">
          <h3>Zone {res["zone"]}</h3>
          <img src="data:image/jpeg;base64,{res["cam1_b64"]}" alt="Camera1"/>
          <img src="data:image/jpeg;base64,{res["cam2_b64"]}" alt="Camera2"/>
          <pre>{res["latest"]}</pre>
        </div>
        """
    html += """
      </div>
    </body>
    </html>
    """
    return html
