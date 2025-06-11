import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv()

import requests
import json
from ai_job_search_app.backend.services.job_search_providers.adzuna_api import search_adzuna_jobs
from ai_job_search_app.backend.services.job_search_providers.theirstack_api import search_theirstack_jobs
from ai_job_search_app.backend.security import create_access_token
from datetime import timedelta

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000"
JOB_SEARCH_ENDPOINT = "/jobs/search"
LOGIN_ENDPOINT = "/auth/login"

# --- User Credentials & Search ---
# !! IMPORTANT !!
# 1. First, run your FastAPI server.
# 2. Use these credentials with the /docs UI to log in and get a token.
# 3. Paste the access token you receive into the `YOUR_ACCESS_TOKEN` variable below.
USER_EMAIL = "test@example.com"  # Use an email you registered with
USER_PASSWORD = "password"       # Use the corresponding password

# This is a temporary addition to generate a token for testing
def get_test_token():
    # Create a dummy user object for token creation
    # In a real app, this user would come from your database
    fake_user = {"username": "testuser", "email": "test@example.com"}
    access_token_expires = timedelta(minutes=30)
    # The create_access_token function will now use the SECRET_KEY from the .env file
    access_token = create_access_token(
        data={"sub": fake_user["username"]}, expires_delta=access_token_expires
    )
    return access_token

# --- !! PASTE YOUR TOKEN HERE !! ---
YOUR_ACCESS_TOKEN = get_test_token()
# ------------------------------------

# --- Search Parameters ---
SEARCH_KEYWORD = "Software Engineer"
SEARCH_LOCATION = "London" # Optional, can be removed to test auto-detection
# -------------------------


def test_job_search():
    """
    Tests the job search endpoint with authentication.
    """
    if YOUR_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("================================================================================")
        print("!! PLEASE UPDATE THE `YOUR_ACCESS_TOKEN` VARIABLE IN THIS SCRIPT !!")
        print("================================================================================")
        return

    headers = {
        "Authorization": f"Bearer {YOUR_ACCESS_TOKEN}"
    }

    params = {
        "keyword": SEARCH_KEYWORD,
        "location": SEARCH_LOCATION
    }
    
    # To test location auto-detection, comment out the line above and uncomment this:
    # params = { "keyword": SEARCH_KEYWORD }

    print(f"[*] Making request to: {BASE_URL}{JOB_SEARCH_ENDPOINT}")
    print(f"[*] With parameters: {params}\n")

    try:
        response = requests.get(
            f"{BASE_URL}{JOB_SEARCH_ENDPOINT}",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        
        print(f"[*] Response Status Code: {response.status_code}")
        print("[*] Response JSON:")
        # Pretty print the JSON response
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\n[!] An error occurred: {e}")
        if e.response:
            print(f"[!] Response Status Code: {e.response.status_code}")
            print(f"[!] Response Body: {e.response.text}")


if __name__ == "__main__":
    test_job_search() 