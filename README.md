# AI Job Search Automation

A FastAPI-based application designed to automate the job search process. It leverages AI to parse CVs, match candidates with job postings from multiple sources, and streamline the application process.

## Current Progress

Implemented and tested the core backend functionality for job searching. Key milestones achieved include:

- **FastAPI Backend**: A robust server has been set up using FastAPI.
- **User Authentication**: Secure user registration and login endpoints (`/auth/register`, `/auth/login`) have been implemented using JWT for token-based authentication.
- **Environment Configuration**: The application now uses a `.env` file to securely manage all secrets and API keys.
- **Database Setup**: Initial database setup with SQLAlchemy is in place to manage users.
- **Job Search Endpoint**: A protected `/jobs/search` endpoint has been created. It aggregates job listings from multiple APIs.
- **API Integration**:
  - **Adzuna API**: Successfully integrated to fetch job listings.
  - **TheirStack API**: Successfully integrated to fetch job listings.
- **Geolocation**: The search endpoint can automatically detect a user's location based on their IP address if one is not provided.
- **End-to-End Testing**: A test script (`test_job_search_endpoint.py`) has been developed and used to successfully validate the entire authentication and job search pipeline.

The core job search API pipeline is now fully functional and tested.