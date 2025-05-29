# Lightbar Monitor Service

## Purpose

This service monitors specified lightbar APIs during a defined blackout period (21:00 - 09:00 America/Edmonton time). If it detects that the lights are on during this period, it attempts to turn them off by sending a command to the lightbar API and sends an alert using Weights & Biases (W&B).

The service checks the light status every minute.

## Configuration

The service is configured using environment variables and a mounted `~/.netrc` file for W&B authentication:

*   **`LIGHTBAR_API_BASE_URLS`** (Required): A comma-separated list of base URLs for the lightbar API endpoints to monitor and control. The script appends `/action/latest` for status checks and `/action` for control.
    *   Example: `http://mitacs-zone1.ccis.ualberta.ca:8080,http://mitacs-zone2.ccis.ualberta.ca:8080,http://mitacs-zone3.ccis.ualberta.ca:8080,http://mitacs-zone6.ccis.ualberta.ca:8080,http://mitacs-zone8.ccis.ualberta.ca:8080,http://mitacs-zone9.ccis.ualberta.ca:8080`
*   **W&B Authentication**: Achieved by mounting your host's `~/.netrc` file (which should contain your W&B credentials) into the container at `/root/.netrc`. Ensure your `~/.netrc` file is correctly configured for W&B access. Example `~/.netrc` entry:
    ```
    machine api.wandb.ai
      login user
      password YOUR_WANDB_API_KEY
    ```
*   **`LIGHT_OFF_PAYLOAD`** (Optional): A JSON string representing the payload to send to the lightbar's `/action` endpoint to turn the lights off.
    *   Default: `{"array": [[0,0,0,0,0,0], [0,0,0,0,0,0]]}`
*   **`WANDB_PROJECT`** (Optional): The W&B project name where alerts will be sent.
    *   Default: `lightbar_monitor_alerts`
*   **`WANDB_ALERT_TITLE`** (Optional): The base title for W&B alerts. The specific URL that triggered the alert will be appended to this title.
    *   Default: `Lights Turned Off During Blackout`
*   **`TZ`** (Timezone): While not directly read as an environment variable by the script for timezone calculation (it uses "America/Edmonton" hardcoded as default for `pytz`), the underlying operating system or Python environment might use `TZ`. For consistent behavior, ensure the environment reflects the intended operational timezone. The script's internal logic for blackout periods is based on "America/Edmonton".

## Running the Service

This service is designed to be run using Docker and Docker Compose. It is included in the main `docker-compose.yml` file at the root of the project.

1.  **Ensure Docker is running.**
2.  **Configure W&B Authentication:**
    *   Make sure your `~/.netrc` file on the Docker host machine is configured with your W&B API key.
3.  **Configure Service in `docker-compose.yml`:**
    *   Modify the `docker-compose.yml` file at the project root to set the `LIGHTBAR_API_BASE_URLS` for the `lightbar-monitor` service with your actual comma-separated URLs.
        ```yaml
        services:
          lightbar-monitor:
            build:
              context: ./lightbar_monitor
              dockerfile: Dockerfile
            networks:
              - plant-rl
            environment:
              # Set your specific lightbar URLs here
              - LIGHTBAR_API_BASE_URLS="http://mitacs-zone1.ccis.ualberta.ca:8080,http://mitacs-zone2.ccis.ualberta.ca:8080,http://mitacs-zone3.ccis.ualberta.ca:8080,http://mitacs-zone6.ccis.ualberta.ca:8080,http://mitacs-zone8.ccis.ualberta.ca:8080,http://mitacs-zone9.ccis.ualberta.ca:8080"
              # ... other optional environment variables like WANDB_PROJECT ...
            volumes:
              # Mounts .netrc from host for W&B authentication
              - ~/.netrc:/root/.netrc:ro 
            restart: unless-stopped
        ```
4.  **Start the service (along with others):**
    ```bash
    docker-compose up -d lightbar-monitor
    ```
    Or, to bring up all services defined in the `docker-compose.yml`:
    ```bash
    docker-compose up -d
    ```
5.  **To view logs:**
    ```bash
    docker-compose logs -f lightbar-monitor
    ```

## How it Works

-   Every 60 seconds, the script checks the current time.
-   If the time is within the blackout period (21:00 - 08:59 America/Edmonton time):
    -   It queries the `/action/latest` endpoint of each specified `LIGHTBAR_API_BASE_URLS`.
    -   If the response indicates lights are on (sum of values in the `action` or `safe_action` array > 0.01):
        -   It sends the `LIGHT_OFF_PAYLOAD` to the `/action` endpoint of that lightbar API.
        -   It sends an alert to the specified W&B project (authenticating via `~/.netrc`).
```
