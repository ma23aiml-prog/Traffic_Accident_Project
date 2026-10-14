
import json
import time
from pathlib import Path

import pandas as pd
import requests

GRID_FILE = Path("data/staging/weather_grid_locations.csv")
RAW_DIR = Path("data/raw/weather")
LOG_DIR = Path("logs")

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

grid = pd.read_csv(GRID_FILE)
session = requests.Session()

success_count = 0
failed_requests = []

for index, row in grid.iterrows():
    latitude = float(row["grid_latitude"])
    longitude = float(row["grid_longitude"])

    filename = f"weather_{latitude:.2f}_{longitude:.2f}.json"
    output_file = RAW_DIR / filename

    # Skip locations already downloaded successfully.
    if output_file.exists():
        try:
            existing = json.loads(output_file.read_text(encoding="utf-8"))
            if len(existing.get("daily", {}).get("time", [])) == 365:
                success_count += 1
                continue
        except (json.JSONDecodeError, OSError):
            pass

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": (
            "temperature_2m_mean,"
            "precipitation_sum,"
            "wind_speed_10m_max"
        ),
        "timezone": "Europe/London",
    }

    try:
        
        # Retry rate-limited and temporary server errors
        data = None

        for attempt in range(5):
            response = session.get(
                API_URL,
                params=params,
                timeout=60
            )

            if response.status_code == 429 or response.status_code >= 500:
                wait_seconds = min(30 * (2 ** attempt), 300)
                print(
                    f"HTTP {response.status_code}. "
                    f"Waiting {wait_seconds} seconds before retry..."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            data = response.json()
            break

        if data is None:
            raise RuntimeError("Weather API failed after 5 attempts")


        days = data.get("daily", {}).get("time", [])
        if len(days) != 365:
            raise ValueError(f"Expected 365 days, received {len(days)}")

        # Preserve the complete raw API response.
        output_file.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        success_count += 1

        if success_count % 25 == 0:
            print(f"Successful locations: {success_count}/{len(grid)}")

        # Be polite to the API.
        time.sleep(2)

    except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
        failed_requests.append({
            "latitude": latitude,
            "longitude": longitude,
            "error": str(error),
        })
        print(f"Failed at {latitude}, {longitude}: {error}")

        # Pause briefly before moving to the next location.
        time.sleep(1)

# Save a failure log for traceability.
failure_file = LOG_DIR / "weather_fetch_failures.json"
failure_file.write_text(
    json.dumps(failed_requests, indent=2),
    encoding="utf-8",
)

print("\n========== WEATHER FETCH SUMMARY ==========")
print("Grid locations:", len(grid))
print("Successful locations:", success_count)
print("Failed requests:", len(failed_requests))
print("Raw weather directory:", RAW_DIR)
print("Failure log:", failure_file)
