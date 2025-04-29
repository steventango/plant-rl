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
      pre {
        white-space: pre-wrap;
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
    </head>
    <body>
      <div class="grid">
    """
    for zone in zones:
        response += f"""
        <div class="zone">
          <h3>Zone {zone}</h3>
          <div class="images">
            <img/>
            <img/>
          </div>
          <pre><code></code></pre>
          <code>Fetched at: <span class="fetch-time" data-iso=""></span></code>
        </div>
        """
    response += """
      </div>
      <script>
        async function reloadZone(zone) {
          const fetchTimeElement = zone.querySelector(".fetch-time");
          fetchTimeElement.textContent = new Date().toLocaleString();
          const images = zone.querySelectorAll("img");
          const zoneId = zone.querySelector("h3").textContent.split(" ")[1];
          for (const [index, img] of images.entries()) {
            const src = "/proxy/zone/0" + zoneId + "/camera/" + (index + 1).toString().padStart(2, "0") + ".jpg";
            const newSrc = src + "?t=" + Date.now();
            const newImg = new Image();
            newImg.onload = () => {
              img.src = newSrc;
            };
            newImg.src = newSrc;
          }
          try {
            const response = await fetch(`/proxy/zone/${zoneId}/latest`);
            const data = await response.json();
            const latestElement = zone.querySelector("pre code");
            const action = data.action.map(arr => arr.map(num => parseFloat(num).toFixed(3)));
            const safeAction = data.safe_action.map(arr => arr.map(num => parseFloat(num).toFixed(3)));
            latestElement.textContent = `Action: [\n  ${action[0]},\n  ${action[1]}\n]\nSafe Action: [\n  ${safeAction[0]},\n  ${safeAction[1]}\n]`;
          } catch (error) {
            const latestElement = zone.querySelector("code");
            const data = await response.text();
            latestElement.textContent = `Error: ${data}`;
          }
        }

        async function reload() {
          const zones = document.querySelectorAll(".zone");
          const reloadPromises = Array.from(zones).map(reloadZone);
          await Promise.all(reloadPromises);
        }
        setInterval(reload, 30000);
        window.onload = reload;
      </script>
    </body>
    </html>
    """
    return response


@app.get("/proxy/zone/{zone}/latest")
async def proxy_latest(zone: int):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://mitacs-zone{zone}.ccis.ualberta.ca:8080/action/latest", timeout=5)
        return Response(content=resp.text, status_code=resp.status_code, media_type=resp.headers.get("Content-Type"))


@app.get("/proxy/zone/{zone}/camera/{camera_id}.jpg")
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
