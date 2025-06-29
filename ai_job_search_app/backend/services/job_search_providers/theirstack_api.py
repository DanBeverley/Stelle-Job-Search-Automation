import requests
import logging
from typing import List, Dict, Any
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

THEIRSTACK_API_KEY = settings.theirstack_api_key
THEIRSTACK_API_URL = settings.theirstack_api_url

def search_theirstack_jobs(keyword: str, location: str) -> List[Dict[str, Any]]:
    """
    Searches for jobs using the TheirStack API and returns them in a standardized format.
    """
    if not THEIRSTACK_API_KEY:
        logger.warning("TheirStack API key not set. Skipping search.")
        return []

    headers = {
        "Authorization": f"Bearer {THEIRSTACK_API_KEY}",
        "Content-Type": "application/json",
    }
    
    body = {
        "order_by": [{"field": "date_posted", "desc": True}],
        "page": 0,
        "limit": 25,
        "job_title_or": [keyword],
        "location_unstructured_or": [location]
    }

    try:
        response = requests.post(THEIRSTACK_API_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        
        standardized_jobs = []
        # Assuming the jobs are in a 'jobs' key in the response
        for job in data.get('jobs', []):
            standardized_jobs.append({
                "title": job.get('job_title'),
                "company": job.get('company_name'),
                "location": ", ".join(job.get('locations_unstructured', [])),
                "description": job.get('job_description_html'), # Assuming HTML description
                "source": "TheirStack"
            })
        return standardized_jobs

    except requests.exceptions.RequestException as e:
        logger.error("TheirStack API request failed: %s", str(e))
        return [] 