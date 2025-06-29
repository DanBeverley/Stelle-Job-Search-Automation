from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

import schemas
from models.db.database import get_db
from models.db import user as user_model
from api.auth import get_current_active_user
from utils.geolocation import get_location_from_ip
from services.job_search_providers import adzuna_api, theirstack_api

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/search", response_model=schemas.JobSearchResult)
def search_jobs(
    request: Request,
    keyword: str,
    location: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Searches for jobs from multiple providers based on a keyword and location.

    - If location is not provided, it attempts to use the user's saved location.
    - If the user has no saved location, it uses IP-based geolocation.
    - The results from all job providers are aggregated.
    """
    logger.info("Starting job search for user: %s", current_user.email)
    final_location = location
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    logger.debug("Current user: %s, DB user found: %s", current_user.email, bool(db_user))

    # 1. Determine the location
    logger.debug("Determining location for job search")
    if not final_location:
        if db_user and db_user.city and db_user.country:
            final_location = f"{db_user.city}, {db_user.country}"
            logger.debug("Using saved user location: %s", final_location)
        else:
            # Fallback to IP geolocation
            client_ip = request.client.host
            logger.debug("No location provided, using IP geolocation for IP: %s", client_ip)
            geo_location = get_location_from_ip(client_ip)
            if geo_location and geo_location.get("city"):
                final_location = f"{geo_location['city']}, {geo_location['country']}"
                logger.info("Location detected from IP: %s", final_location)
                # 2. Update user profile with the new location
                if db_user:
                    db_user.city = geo_location['city']
                    db_user.country = geo_location['country']
                    db.commit()
                    logger.debug("Updated user location in database")
    logger.info("Final location determined: %s", final_location)

    if not final_location:
        logger.warning("Location could not be determined for user: %s", current_user.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine location. Please provide one manually.",
        )

    # 3. Call job provider services
    try:
        logger.debug("Calling Adzuna API for keyword: %s, location: %s", keyword, final_location)
        adzuna_jobs_raw = adzuna_api.search_adzuna_jobs(keyword, final_location)
        logger.info("Adzuna API returned %d jobs", len(adzuna_jobs_raw))
    except Exception as e:
        logger.error("Adzuna API call failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Adzuna API Error: {e}")

    try:
        logger.debug("Calling TheirStack API for keyword: %s, location: %s", keyword, final_location)
        theirstack_jobs_raw = theirstack_api.search_theirstack_jobs(keyword, final_location)
        logger.info("TheirStack API returned %d jobs", len(theirstack_jobs_raw))
    except Exception as e:
        logger.error("TheirStack API call failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"TheirStack API Error: {e}")

    # 4. Aggregate and format results
    logger.debug("Aggregating and formatting job results")
    all_jobs: List[schemas.JobListing] = []

    # This is where normalize the raw data from each API
    # into the standard `JobListing` schema.
    all_jobs.extend([schemas.JobListing(**job) for job in adzuna_jobs_raw])
    all_jobs.extend([schemas.JobListing(**job) for job in theirstack_jobs_raw])
    logger.info("Job search completed. Returning %d total jobs", len(all_jobs))
    
    return {"jobs": all_jobs} 