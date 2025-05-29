import os
import time
import datetime
import pytz
import requests
import wandb
import numpy as np
import json

def get_env_var(var_name, default_value=None):
    """Gets an environment variable or returns a default value."""
    value = os.getenv(var_name)
    if value is None:
        if default_value is None:
            print(f"Error: Environment variable {var_name} not set and no default value provided.")
            # For LIGHTBAR_API_BASE_URLS, we'll handle the error in main if it's empty after splitting
            if var_name != "LIGHTBAR_API_BASE_URLS": 
                exit(1)
        return default_value
    return value

def is_blackout_period(timezone_str="America/Edmonton", blackout_start_hour=21, blackout_end_hour=9):
    """Checks if the current time is within the blackout period for the given timezone."""
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Unknown timezone '{timezone_str}'. Exiting.")
        # This is a critical error, so exit.
        exit(1) 
    now_local = datetime.datetime.now(tz)
    current_hour = now_local.hour
    # Blackout is from start_hour (e.g., 21) up to (but not including) end_hour (e.g., 9 the next day)
    if blackout_start_hour <= current_hour or current_hour < blackout_end_hour:
        return True
    return False

def get_light_status(api_base_url):
    """Checks if the lights are currently on by querying the lightbar API."""
    status_url = f"{api_base_url}/action/latest"
    print(f"Checking light status for {api_base_url} at {status_url}")
    try:
        response = requests.get(status_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        action = data.get("action")
        safe_action = data.get("safe_action")
        current_action_to_check = action if action is not None else safe_action
        if current_action_to_check is None:
            print(f"Warning: Both 'action' and 'safe_action' are null from {status_url}.")
            return False
        action_array = np.array(current_action_to_check)
        if np.sum(action_array) > 0.01:
            print(f"Lights are ON for {api_base_url}")
            return True
        print(f"Lights are OFF for {api_base_url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error fetching light status from {status_url}: {e}")
        return False
    except (ValueError, KeyError) as e:
        print(f"Error parsing light status JSON from {status_url}: {e}")
        return False

def turn_off_lights(api_base_url, payload_str):
    """Turns off the lights using the lightbar API."""
    control_url = f"{api_base_url}/action"
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        print(f"Error: LIGHT_OFF_PAYLOAD is not valid JSON: {payload_str}")
        print("Using default payload...")
        payload = {"array": [[0,0,0,0,0,0], [0,0,0,0,0,0]]}
    try:
        response = requests.put(control_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Successfully sent request to turn off lights to {control_url}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error turning off lights via {control_url}: {e}")
        return False

def send_wandb_alert(project, title, text, api_key):
    """Sends an alert using Weights & Biases."""
    if not api_key:
        print("Error: WANDB_API_KEY not set. Cannot send alert.")
        return False
    try:
        # Attempt to login. If it fails, wandb.login() might raise an error or return False/None.
        # The wandb library handles login state internally. Explicit login here ensures credentials are used.
        if not wandb.login(key=api_key):
             print("W&B login failed. Please check API key.")
             return False

        run = wandb.init(project=project, job_type="alert_sender", reinit=True)
        if run:
            run.alert(title=title, text=text)
            run.finish()
            print(f"W&B alert sent: '{title}' for text: '{text}'")
            return True
        else:
            print("Error: Failed to initialize W&B run for alert.")
            return False
    except Exception as e:
        print(f"Error sending W&B alert: {e}")
        return False

def main():
    print("Starting Light Monitor Service...")
    
    wandb_api_key = get_env_var("WANDB_API_KEY")
    lightbar_api_base_urls_str = get_env_var("LIGHTBAR_API_BASE_URLS") 
    
    default_light_off_payload = '{"array": [[0,0,0,0,0,0], [0,0,0,0,0,0]]}'
    light_off_payload_str = get_env_var("LIGHT_OFF_PAYLOAD", default_light_off_payload)
    
    wandb_project = get_env_var("WANDB_PROJECT", "lightbar_monitor_alerts")
    default_wandb_alert_title = "Lights Turned Off During Blackout"
    
    if not lightbar_api_base_urls_str:
        print("Error: LIGHTBAR_API_BASE_URLS environment variable not set or empty.")
        exit(1)

    base_urls = [url.strip() for url in lightbar_api_base_urls_str.split(',') if url.strip()]

    if not base_urls:
        print("Error: No valid URLs found in LIGHTBAR_API_BASE_URLS after parsing.")
        exit(1)

    print(f"Monitoring lights for URLs: {base_urls}")
    # Corrected print message to reflect that blackout_end_hour is exclusive
    print(f"Blackout period: 21:00 - <09:00 America/Edmonton") 
    print(f"W&B Project for alerts: {wandb_project}")

    while True:
        print(f"[{datetime.datetime.now()}] Checking conditions for all URLs...")
        # Blackout period check is done once for all URLs
        if is_blackout_period():
            print("Current time is within the blackout period.")
            for base_url in base_urls:
                # Basic URL validation
                if not base_url.startswith("http://") and not base_url.startswith("https://"):
                    print(f"Warning: URL '{base_url}' does not start with http:// or https://. Skipping.")
                    continue

                print(f"--- Processing URL: {base_url} ---")
                lights_are_on = get_light_status(base_url)
                if lights_are_on:
                    # This print is already inside get_light_status
                    # print(f"Lights are ON for {base_url}. Attempting to turn them off.") 
                    if turn_off_lights(base_url, light_off_payload_str):
                        alert_title_template = get_env_var("WANDB_ALERT_TITLE", default_wandb_alert_title)
                        # Make title specific to the URL
                        alert_title_specific = f"{alert_title_template}: {base_url}" 
                        alert_text = (f"Lights were detected ON and turned OFF "
                                      f"during the blackout period for {base_url}.")
                        send_wandb_alert(wandb_project, alert_title_specific, alert_text, wandb_api_key)
                # else: # No need for this else, get_light_status prints if lights are off
                    # print(f"Lights are OFF for {base_url} or status could not be determined.")
                print(f"--- Finished processing URL: {base_url} ---")
        else:
            print("Current time is outside the blackout period. No action needed for any URL.")
        
        print(f"Waiting for 60 seconds before next check cycle for all URLs...")
        time.sleep(60)

if __name__ == "__main__":
    main()
