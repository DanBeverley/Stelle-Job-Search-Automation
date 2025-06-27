import requests
import logging
from typing import List, Dict, Any
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

ADZUNA_APP_ID = settings.adzuna_app_id
ADZUNA_APP_KEY = settings.adzuna_app_key
ADZUNA_API_URL = settings.adzuna_api_url

def search_adzuna_jobs(keyword: str, location: str) -> List[Dict[str, Any]]:
    """
    Searches for jobs on Adzuna and returns them in a standardized format.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        logger.warning("Adzuna API credentials not set. Skipping search.")
        return []

    params = {
        'app_id': ADZUNA_APP_ID,
        'app_key': ADZUNA_APP_KEY,
        'results_per_page': 20,
        'what': keyword,
        'where': location,
        'content-type': 'application/json'
    }

    try:
        response = requests.get(ADZUNA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Normalize the response to our standard JobListing format
        standardized_jobs = []
        for job in data.get('results', []):
            standardized_jobs.append({
                "title": job.get('title'),
                "company": job.get('company', {}).get('display_name'),
                "location": job.get('location', {}).get('display_name'),
                "description": job.get('description'),
                "source": "Adzuna"
            })
        return standardized_jobs

    except requests.exceptions.RequestException as e:
        logger.error("Adzuna API request failed: %s", str(e))
        return [] 