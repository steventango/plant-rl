import asyncio
import html
import json
from datetime import datetime

import httpx
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def show_zones():
    zones = [1, 2, 3, 6, 8, 9]

    async def fetch_zone_data(zone):
        result = {}
        result["zone"] = zone
        async with httpx.AsyncClient() as client:
            try:
                latest_resp = await client.get(
                    f"http://mitacs-zone{zone}.ccis.ualberta.ca:8080/action/latest", timeout=5
                )
                latest_resp.raise_for_status()
                latest_data = latest_resp.json()
                for key, value in latest_data.items():
                    latest_data[key] = np.round(value, 3).tolist()
                result["latest"] = f"Action:[\n  {latest_data['action'][0]},\n  {latest_data['action'][1]}\n]\n"
                result[
                    "latest"
                ] += f"Safe Action: [\n  {latest_data['safe_action'][0]},\n  {latest_data['safe_action'][1]}\n]"
            except Exception as e:
                result["latest"] = f"Error: {e}"
        result["fetch_time"] = datetime.now().isoformat()
        return result

    tasks = [fetch_zone_data(zone) for zone in zones]
    results = await asyncio.gather(*tasks)

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
        border: 1px solid #000;
        height: 50vh;
        overflow: hidden;
        display: grid;
        grid-template-rows: auto;
        align-content: start;
      }
      h3 {
        text-align: center;
        grid-row: 1 / 2;
        margin: 0;
      }
      body {
        font-family: sans-serif;
        margin: 0;
      }
      img {
        width: 100%;
        aspect-ratio: 4 / 3;
        background-color: #ccc;
        grid-row: 2 / 3;
      }
      .images {
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-row: 2 / 3;
      }
    </style>
    <script>
      function convertToLocalTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString();
      }
      document.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll(".fetch-time").forEach(element => {
          const isoString = element.getAttribute("data-iso");
          element.textContent = convertToLocalTime(isoString);
        });
      });
    </script>
    <script>
      function reloadImages() {
        document.querySelectorAll(".zone").forEach(zone => {
          zone.querySelectorAll("img").forEach(img => {
            const src = img.getAttribute("src");
            const newSrc = src.split("?")[0] + "?" + new Date().getTime();
            const newImg = new Image();
            newImg.onload = () => {
              img.src = newSrc;
            };
            newImg.src = newSrc;
          });
          const fetchTimeElement = zone.querySelector(".fetch-time");
          if (fetchTimeElement) {
            fetchTimeElement.textContent = convertToLocalTime(new Date().toISOString());
          }
        });
      }
      setInterval(reloadImages, 30000);
    </script>
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
          <div class="images">
            <img src="/proxy/zone/{res['zone']}/camera/1.png"/>
            <img src="/proxy/zone/{res['zone']}/camera/2.png"/>
          </div>
          <br>
          <pre><code>{code}</code></pre>
          <br>
          <code>Fetched at: <span class="fetch-time" data-iso="{fetch_time}"></span></code>
        </div>
        """
    response += """
      </div>
    </body>
    </html>
    """
    return response


@app.get("/proxy/zone/{zone}/latest")
async def proxy_latest(zone: int):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca:8080/action/latest", timeout=5)
        return Response(content=resp.text, status_code=resp.status_code, media_type=resp.headers.get("Content-Type"))


@app.get("/proxy/zone/{zone}/camera/{camera_id}.png")
async def proxy_camera(zone: int, camera_id: int):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"http://mitacs-zone{zone:02d}-camera{camera_id:02d}.ccis.ualberta.ca:8080/observation",
            timeout=30,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp.headers,
            media_type=resp.headers.get("Content-Type"),
        )
