import requests
from typing import List, Dict, Any
import os
import re
from cachetools import cached, TTLCache

# --- edX API Configuration ---
EDX_API_CLIENT_ID = os.environ.get("EDX_API_CLIENT_ID")
EDX_API_CLIENT_SECRET = os.environ.get("EDX_API_CLIENT_SECRET")
EDX_TOKEN_URL = "https://api.edx.org/oauth2/v1/access_token"
EDX_COURSE_API_URL = "https://api.edx.org/catalog/v1/courses"

# Define a common list of technical skills for extraction
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'javascript', 'typescript', 'sql', 'nosql', 'react', 'vue', 'angular', 
    'node.js', 'fastapi', 'flask', 'django', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
    'scikit-learn', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'machine learning', 'data analysis',
    'project management', 'agile', 'scrum', 'ui/ux', 'figma', 'adobe xd'
]

# Cache the access token for 1 hour (3600 seconds)
token_cache = TTLCache(maxsize=1, ttl=3600)

@cached(token_cache)
def get_edx_access_token() -> str:
    """
    Retrieves an access token from the edX API using client credentials.
    The token is cached to avoid repeated requests.
    """
    payload = {
        'grant_type': 'client_credentials',
        'client_id': EDX_API_CLIENT_ID,
        'client_secret': EDX_API_CLIENT_SECRET,
        'token_type': 'jwt'
    }
    response = requests.post(EDX_TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json().get('access_token')

def get_course_recommendations(skill: str) -> List[Dict[str, Any]]:
    """
    Fetches course recommendations for a specific skill from the edX API.
    """
    try:
        access_token = get_edx_access_token()
        headers = {'Authorization': f'JWT {access_token}'}
        params = {'search_query': skill, 'size': 5} 
        
        response = requests.get(EDX_COURSE_API_URL, headers=headers, params=params)
        response.raise_for_status()
        
        courses = []
        for course in response.json().get('results', []):
            courses.append({
                "title": course.get('title'),
                "url": course.get('marketing_url'),
                "organization": course.get('owners', [{}])[0].get('name', 'N/A')
            })
        return courses
    except requests.exceptions.RequestException as e:
        print(f"Error fetching courses for '{skill}': {e}")
        return []

def extract_skills_from_description(description: str) -> List[str]:
    """
    Extracts predefined keywords from a job description using regex.
    """
    found_skills = set()
    for skill in SKILL_KEYWORDS:
        # Use word boundaries to match whole words and be case-insensitive
        if re.search(r'\b' + re.escape(skill) + r'\b', description, re.IGNORECASE):
            found_skills.add(skill)
    return list(found_skills)

def analyze_skill_gap(job_description: str, user_skills: List[str]) -> Dict[str, Any]:
    """
    Analyzes the gap between user skills and job requirements, and provides course recommendations.
    """
    required_skills = extract_skills_from_description(job_description)
    user_skills_set = set(skill.lower() for skill in user_skills)
    
    matched_skills = [skill for skill in required_skills if skill.lower() in user_skills_set]
    missing_skills = [skill for skill in required_skills if skill.lower() not in user_skills_set]
    
    recommended_courses = []
    # Get recommendations for the first few missing skills to avoid too many API calls
    for skill in missing_skills[:3]:
        recommended_courses.extend(get_course_recommendations(skill))
        
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommended_courses": recommended_courses
    } 