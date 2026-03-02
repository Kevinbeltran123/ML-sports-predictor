"""Open-Meteo weather provider for MLB stadium conditions.

Free API, no authentication required, no rate limits.
  Forecast:   https://api.open-meteo.com/v1/forecast
  Historical: https://archive-api.open-meteo.com/v1/archive

Conversions:
  Temperature: Celsius → Fahrenheit
  Wind speed:  km/h → mph

Cache: same-day lookups are cached in memory via functools.lru_cache.
Default fallback: 72 F, 5 mph, 0 deg, 50% humidity, 0% precip probability.
"""

import logging
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

_FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"
_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly variable names requested from the API
_HOURLY_VARS = (
    "temperature_2m,"
    "windspeed_10m,"
    "winddirection_10m,"
    "relativehumidity_2m,"
    "precipitation_probability"
)

_HOURLY_VARS_ARCHIVE = (
    "temperature_2m,"
    "windspeed_10m,"
    "winddirection_10m,"
    "relativehumidity_2m"
    # precipitation_probability is not available in the archive API
)

# Defaults returned on any API error
_DEFAULTS = {
    "temp_f":        72.0,
    "wind_speed_mph": 5.0,
    "wind_dir_deg":   0.0,
    "humidity_pct":  50.0,
    "precip_prob":    0.0,
}


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371


# ---------------------------------------------------------------------------
# Internal fetch helper
# ---------------------------------------------------------------------------

def _fetch_weather(base_url: str, lat: float, lon: float,
                   date: str, hour: int,
                   hourly_vars: str,
                   include_precip_prob: bool = True) -> dict:
    """Shared fetch logic for forecast and historical endpoints.

    Args:
        base_url:            URL of the Open-Meteo endpoint to query
        lat, lon:            stadium coordinates
        date:                YYYY-MM-DD
        hour:                local hour to extract (0-23)
        hourly_vars:         comma-separated hourly variable names
        include_precip_prob: whether precipitation_probability is in the response

    Returns:
        Weather dict or _DEFAULTS on error.
    """
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "hourly":      hourly_vars,
        "start_date":  date,
        "end_date":    date,
        "timezone":    "America/New_York",  # consistent reference tz; caller passes local hour
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Weather API request failed (lat=%.4f lon=%.4f date=%s): %s",
            lat, lon, date, exc,
        )
        return dict(_DEFAULTS)
    except Exception as exc:
        logger.error("Unexpected weather API error: %s", exc)
        return dict(_DEFAULTS)

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    # Find the index matching the requested hour
    target_idx = None
    for idx, t in enumerate(times):
        # times are "YYYY-MM-DDTHH:00" strings
        try:
            t_hour = int(t.split("T")[1].split(":")[0])
            if t_hour == hour:
                target_idx = idx
                break
        except (IndexError, ValueError):
            continue

    if target_idx is None:
        logger.warning(
            "Hour %d not found in Open-Meteo response for date=%s. Using defaults.",
            hour, date,
        )
        return dict(_DEFAULTS)

    def _safe(key: str, fallback=0.0):
        vals = hourly.get(key, [])
        v = vals[target_idx] if target_idx < len(vals) else None
        return float(v) if v is not None else fallback

    temp_c       = _safe("temperature_2m",   20.0)
    wind_kmh     = _safe("windspeed_10m",     8.0)
    wind_dir     = _safe("winddirection_10m", 0.0)
    humidity     = _safe("relativehumidity_2m", 50.0)
    precip_prob  = _safe("precipitation_probability", 0.0) if include_precip_prob else 0.0

    return {
        "temp_f":         round(_celsius_to_fahrenheit(temp_c), 1),
        "wind_speed_mph": round(_kmh_to_mph(wind_kmh), 1),
        "wind_dir_deg":   round(wind_dir, 1),
        "humidity_pct":   round(humidity, 1),
        "precip_prob":    round(precip_prob, 1),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_weather_forecast(lat: float, lon: float, date: str, hour: int = 19) -> dict:
    """Get weather forecast for a stadium location at game time.

    Results are cached in memory keyed by (lat, lon, date, hour). Calling
    the same arguments multiple times in the same process incurs no extra
    HTTP requests.

    Args:
        lat:  Stadium latitude  (e.g. 40.8296 for Yankee Stadium)
        lon:  Stadium longitude (e.g. -73.9262)
        date: YYYY-MM-DD
        hour: Local hour of game start (default 19 = 7 PM)

    Returns:
        Dict with keys:
          temp_f         - Temperature in Fahrenheit
          wind_speed_mph - Wind speed in mph
          wind_dir_deg   - Wind direction in degrees (0 = N, 90 = E, etc.)
          humidity_pct   - Relative humidity percentage
          precip_prob    - Precipitation probability percentage (0-100)
    """
    logger.debug(
        "Fetching weather forecast: lat=%.4f lon=%.4f date=%s hour=%d",
        lat, lon, date, hour,
    )
    return _fetch_weather(
        base_url=_FORECAST_URL,
        lat=lat,
        lon=lon,
        date=date,
        hour=hour,
        hourly_vars=_HOURLY_VARS,
        include_precip_prob=True,
    )


@lru_cache(maxsize=4096)
def get_historical_weather(lat: float, lon: float, date: str, hour: int = 19) -> dict:
    """Get historical weather for a stadium location (for training data).

    Uses the Open-Meteo archive API which covers dates from 1940 onwards.
    Note: precipitation_probability is not available in the archive; it will
    be returned as 0.0.

    Args:
        lat:  Stadium latitude
        lon:  Stadium longitude
        date: YYYY-MM-DD
        hour: Local hour of game start (default 19 = 7 PM)

    Returns:
        Dict with keys:
          temp_f, wind_speed_mph, wind_dir_deg, humidity_pct, precip_prob
          (same schema as get_weather_forecast; precip_prob always 0.0)
    """
    logger.debug(
        "Fetching historical weather: lat=%.4f lon=%.4f date=%s hour=%d",
        lat, lon, date, hour,
    )
    return _fetch_weather(
        base_url=_HISTORICAL_URL,
        lat=lat,
        lon=lon,
        date=date,
        hour=hour,
        hourly_vars=_HOURLY_VARS_ARCHIVE,
        include_precip_prob=False,
    )


# ---------------------------------------------------------------------------
# Stadium coordinates reference
# ---------------------------------------------------------------------------

# Approximate GPS coordinates for all 30 MLB stadiums.
# Useful for quick lookups without an external geocoding call.
MLB_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    # (latitude, longitude)
    "Arizona Diamondbacks":   (33.4453,  -112.0667),  # Chase Field
    "Atlanta Braves":         (33.8907,   -84.4677),  # Truist Park
    "Baltimore Orioles":      (39.2838,   -76.6217),  # Oriole Park
    "Boston Red Sox":         (42.3467,   -71.0972),  # Fenway Park
    "Chicago Cubs":           (41.9484,   -87.6553),  # Wrigley Field
    "Chicago White Sox":      (41.8300,   -87.6338),  # Guaranteed Rate Field
    "Cincinnati Reds":        (39.0975,   -84.5069),  # Great American Ball Park
    "Cleveland Guardians":    (41.4962,   -81.6853),  # Progressive Field
    "Colorado Rockies":       (39.7559,  -104.9942),  # Coors Field
    "Detroit Tigers":         (42.3390,   -83.0485),  # Comerica Park
    "Houston Astros":         (29.7573,   -95.3555),  # Minute Maid Park
    "Kansas City Royals":     (39.0517,   -94.4803),  # Kauffman Stadium
    "Los Angeles Angels":     (33.8003,  -117.8827),  # Angel Stadium
    "Los Angeles Dodgers":    (34.0739,  -118.2400),  # Dodger Stadium
    "Miami Marlins":          (25.7781,   -80.2197),  # loanDepot park
    "Milwaukee Brewers":      (43.0280,   -87.9712),  # American Family Field
    "Minnesota Twins":        (44.9817,   -93.2775),  # Target Field
    "New York Mets":          (40.7571,   -73.8458),  # Citi Field
    "New York Yankees":       (40.8296,   -73.9262),  # Yankee Stadium
    "Oakland Athletics":      (37.7516,  -122.2005),  # Oakland Coliseum
    "Philadelphia Phillies":  (39.9057,   -75.1665),  # Citizens Bank Park
    "Pittsburgh Pirates":     (40.4469,   -80.0057),  # PNC Park
    "San Diego Padres":       (32.7076,  -117.1570),  # Petco Park
    "San Francisco Giants":   (37.7786,  -122.3893),  # Oracle Park
    "Seattle Mariners":       (47.5914,  -122.3325),  # T-Mobile Park
    "St. Louis Cardinals":    (38.6226,   -90.1928),  # Busch Stadium
    "Tampa Bay Rays":         (27.7683,   -82.6534),  # Tropicana Field (dome)
    "Texas Rangers":          (32.7512,   -97.0832),  # Globe Life Field (dome)
    "Toronto Blue Jays":      (43.6414,   -79.3894),  # Rogers Centre (dome)
    "Washington Nationals":   (38.8730,   -77.0074),  # Nationals Park
}


def get_stadium_coords(team_name: str) -> tuple[float, float] | None:
    """Look up stadium GPS coordinates for a team.

    Args:
        team_name: Full MLB team name (e.g. "New York Yankees")

    Returns:
        (latitude, longitude) tuple, or None if not found.
    """
    return MLB_STADIUM_COORDS.get(team_name)
