import requests
import logging

logger = logging.getLogger(__name__)

def get_location_from_ip(ip_address: str) -> dict:
    """
    Gets location information from an IP address using the ip-api.com service.

    Args:
        ip_address: The IP address to geolocate.

    Returns:
        A dictionary containing location details (e.g., city, country)
        or None if the request fails or the IP is unroutable.
    """
    if ip_address in ["127.0.0.1", "localhost"]:
        return None # Cannot geolocate local addresses

    try:
        response = requests.get(f"http://ip-api.com/json/{ip_address}")
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if data.get("status") == "success":
            return {
                "city": data.get("city"),
                "country": data.get("country"),
            }
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Geolocation request failed: %s", str(e))
        return None 