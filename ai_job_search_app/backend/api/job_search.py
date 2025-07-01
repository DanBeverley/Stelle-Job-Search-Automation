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

@router.get("/auto-search", response_model=schemas.JobSearchResult)
def automated_job_search(
    request: Request,
    location: Optional[str] = None,
    max_jobs: int = 20,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Automated job search based on user's CV skills and preferences.
    Searches for jobs matching the user's skills from their uploaded CV.
    """
    logger.info("Starting automated job search for user: %s", current_user.email)
    
    # Get user with CV data
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    
    if not db_user or not db_user.parsed_cv_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No CV data found. Please upload your CV first to enable automated job search."
        )
    
    cv_data = db_user.parsed_cv_data
    skills = cv_data.get('skills', [])
    
    if not skills:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No skills found in your CV. Please update your CV with relevant skills."
        )
    
    # Determine location
    final_location = location
    if not final_location:
        if db_user.city and db_user.country:
            final_location = f"{db_user.city}, {db_user.country}"
        else:
            client_ip = request.client.host
            geo_location = get_location_from_ip(client_ip)
            if geo_location and geo_location.get("city"):
                final_location = f"{geo_location['city']}, {geo_location['country']}"
    
    if not final_location:
        final_location = "Remote"  # Default fallback
    
    logger.info("Automated search for skills: %s in location: %s", skills[:3], final_location)
    
    all_jobs = []
    search_count = 0
    
    # Search with top skills from CV
    for skill in skills[:5]:  # Use top 5 skills to avoid too many API calls
        if search_count >= 3:  # Limit to prevent API rate limiting
            break
            
        try:
            # Search with individual skills
            adzuna_jobs = adzuna_api.search_adzuna_jobs(skill, final_location)
            theirstack_jobs = theirstack_api.search_theirstack_jobs(skill, final_location)
            
            # Convert to schema objects
            skill_jobs = []
            skill_jobs.extend([schemas.JobListing(**job) for job in adzuna_jobs])
            skill_jobs.extend([schemas.JobListing(**job) for job in theirstack_jobs])
            
            # Add skill relevance info
            for job in skill_jobs:
                job.description = f"[Matched skill: {skill}] {job.description}"
            
            all_jobs.extend(skill_jobs)
            search_count += 1
            
            logger.info("Found %d jobs for skill: %s", len(skill_jobs), skill)
            
        except Exception as e:
            logger.error("Search failed for skill %s: %s", skill, str(e))
            continue
    
    # Remove duplicates based on title and company
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        key = (job.title.lower(), job.company.lower())
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)
    
    # Limit results
    unique_jobs = unique_jobs[:max_jobs]
    
    logger.info("Automated job search completed. Found %d unique jobs", len(unique_jobs))
    
    return {"jobs": unique_jobs}

@router.post("/save-search")
def save_job_search_preferences(
    preferences: dict,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Save user's job search preferences for automation.
    """
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    
    # Store preferences in user data (could extend user model)
    # For now, store in parsed_cv_data as job_preferences
    if not db_user.parsed_cv_data:
        db_user.parsed_cv_data = {}
    
    db_user.parsed_cv_data['job_preferences'] = preferences
    db.commit()
    
    return {"message": "Job search preferences saved successfully"}

@router.get("/test-auto-search")
def test_automated_job_search(location: Optional[str] = "Remote"):
    """
    Test endpoint for automated job search without authentication.
    For testing purposes only.
    """
    try:
        # Mock CV skills for testing
        mock_skills = ["Python", "JavaScript", "React", "FastAPI", "Machine Learning"]
        
        logger.info("Testing automated search for skills: %s in location: %s", mock_skills[:3], location)
        
        all_jobs = []
        search_count = 0
        
        # Search with mock skills
        for skill in mock_skills[:3]:  # Use top 3 skills for testing
            if search_count >= 2:  # Limit to prevent API rate limiting
                break
                
            try:
                # Search with individual skills
                adzuna_jobs = adzuna_api.search_adzuna_jobs(skill, location)
                theirstack_jobs = theirstack_api.search_theirstack_jobs(skill, location)
                
                # Convert to schema objects
                skill_jobs = []
                for job in adzuna_jobs:
                    job_obj = schemas.JobListing(**job)
                    job_obj.description = f"[Matched skill: {skill}] {job_obj.description[:200]}..."
                    skill_jobs.append(job_obj)
                
                for job in theirstack_jobs:
                    job_obj = schemas.JobListing(**job)
                    job_obj.description = f"[Matched skill: {skill}] {job_obj.description[:200]}..."
                    skill_jobs.append(job_obj)
                
                all_jobs.extend(skill_jobs)
                search_count += 1
                
                logger.info("Found %d jobs for skill: %s", len(skill_jobs), skill)
                
            except Exception as e:
                logger.error("Search failed for skill %s: %s", skill, str(e))
                continue
        
        # Remove duplicates
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = (job.title.lower(), job.company.lower())
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        # Limit results
        unique_jobs = unique_jobs[:15]
        
        return {
            "jobs": unique_jobs,
            "skills_searched": mock_skills[:search_count],
            "total_found": len(unique_jobs),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "jobs": [],
            "error": str(e),
            "status": "error"
        } 