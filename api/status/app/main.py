import base64
import concurrent.futures
import html
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
            latest_resp = requests.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca:8080/action/latest", timeout=5)
            latest_resp.raise_for_status()
            result["latest"] = json.dumps(latest_resp.json())
        except Exception as e:
            result["latest"] = f"Error: {e}"
        try:
            cam1_resp = requests.get(f"http://mitacs-zone0{zone}-camera01.ccis.ualberta.ca:8080/observation", timeout=15)
            cam1_resp.raise_for_status()
            result["cam1_b64"] = base64.b64encode(cam1_resp.content).decode()
        except Exception as e:
            result["cam1_b64"] = f"Error: {e}"
        try:
            cam2_resp = requests.get(f"http://mitacs-zone0{zone}-camera02.ccis.ualberta.ca:8080/observation", timeout=15)
            cam2_resp.raise_for_status()
            result["cam2_b64"] = base64.b64encode(cam2_resp.content).decode()
        except Exception as e:
            result["cam2_b64"] = f"Error: {e}"
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
      }
    </style>
    </head>
    <body>
      <div class="grid">
    """
    for res in results:
        code = html.escape(res["latest"])
        cam1_src = f"data:image/png;base64,{res['cam1_b64']}"
        if res["cam1_b64"].startswith("Error:"):
            cam1_src = "https://placehold.co/400x300"
            code += "<br><br>" + html.escape(res["cam1_b64"])
        cam2_src = f"data:image/png;base64,{res['cam2_b64']}"
        if res["cam2_b64"].startswith("Error:"):
            cam2_src = "https://placehold.co/400x300"
            code += "<br><br>" + html.escape(res["cam2_b64"])
        response += f"""
        <div class="zone">
          <h3>Zone {res["zone"]}</h3>
          <img src="{cam1_src}"/>
          <img src="{cam2_src}"/>
          <br>
          <code>{code}</code>
        </div>
        """
    response += """
      </div>
    </body>
    </html>
    """
    return response
