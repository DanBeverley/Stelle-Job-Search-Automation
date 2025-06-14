from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from typing import Optional, List

from .. import schemas
from ..models.db.database import get_db
from ..models.db import user as user_model
from ..api.auth import get_current_active_user
from ..utils.geolocation import get_location_from_ip
from ..services.job_search_providers import adzuna_api, theirstack_api

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
    print("[DEBUG] Entered search_jobs endpoint.")
    final_location = location
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    print(f"[DEBUG] Current user: {current_user.email}, DB user found: {'Yes' if db_user else 'No'}")

    # 1. Determine the location
    print("[DEBUG] Determining location...")
    if not final_location:
        if db_user and db_user.city and db_user.country:
            final_location = f"{db_user.city}, {db_user.country}"
        else:
            # Fallback to IP geolocation
            client_ip = request.client.host
            print(f"[DEBUG] No location provided, using IP geolocation for IP: {client_ip}")
            geo_location = get_location_from_ip(client_ip)
            if geo_location and geo_location.get("city"):
                final_location = f"{geo_location['city']}, {geo_location['country']}"
                # 2. Update user profile with the new location
                if db_user:
                db_user.city = geo_location['city']
                db_user.country = geo_location['country']
                db.commit()
    print(f"[DEBUG] Final location determined: {final_location}")

    if not final_location:
        print("[DEBUG] Location could not be determined. Raising HTTPException.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine location. Please provide one manually.",
        )

    # 3. Call job provider services
    try:
        print("[DEBUG] Calling Adzuna API...")
    adzuna_jobs_raw = adzuna_api.search_adzuna_jobs(keyword, final_location)
        print("[DEBUG] Adzuna API call successful.")
    except Exception as e:
        print(f"[DEBUG] Adzuna API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Adzuna API Error: {e}")

    try:
        print("[DEBUG] Calling TheirStack API...")
    theirstack_jobs_raw = theirstack_api.search_theirstack_jobs(keyword, final_location)
        print("[DEBUG] TheirStack API call successful.")
    except Exception as e:
        print(f"[DEBUG] TheirStack API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"TheirStack API Error: {e}")

    # 4. Aggregate and format results
    print("[DEBUG] Aggregating and formatting results...")
    all_jobs: List[schemas.JobListing] = []

    # This is where you would normalize the raw data from each API
    # into the standard `JobListing` schema. For now, we assume the
    # provider modules will eventually return data in this format.
    all_jobs.extend([schemas.JobListing(**job) for job in adzuna_jobs_raw])
    all_jobs.extend([schemas.JobListing(**job) for job in theirstack_jobs_raw])
    print("[DEBUG] Formatting complete. Returning results.")
    
    return {"jobs": all_jobs} 