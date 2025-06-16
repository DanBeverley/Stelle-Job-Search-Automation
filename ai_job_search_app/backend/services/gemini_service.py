import os
import google.generativeai as genai
import ast
import json
import io
from PIL import Image
import logging
import pillow_avif  
import requests


logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")


def generate_interview_questions_with_gemini(job_description: str, cv_text: str = "") -> list[str]:
    """
    Uses the Gemini API to generate interview questions based on a job description and CV.

    Args:
        job_description: The full text of the job description.
        cv_text: The full text of the user's CV.

    Returns:
        A list of generated interview questions.
    """
    # Construct a detailed prompt for the model
    prompt = f"""
    Act as an expert technical recruiter and career coach. Your task is to generate a list of 5 to 7 insightful interview questions.

    These questions should be based on the provided job description and the candidate's CV. They should probe the candidate's skills, experience, and suitability for the role described.

    The questions should be a mix of:
    1.  Behavioral questions ('Tell me about a time when...').
    2.  Technical questions related to the skills mentioned in the job description.
    3.  Situational questions ('How would you handle...').

    Return the questions as a Python list of strings. For example: ["Question 1?", "Question 2?"]

    ---
    JOB DESCRIPTION:
    {job_description}
    ---
    CANDIDATE CV:
    {cv_text}
    ---
    INTERVIEW QUESTIONS:
    """

    try:
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        
        questions = [q.strip() for q in response_text.split('\n') if q.strip()]
        
        if not questions:
            return ["Error: Model returned an empty response."]
            
        return questions

    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        # In case of any error (API, parsing, etc.), return a helpful error message.
        return [f"Failed to generate questions due to an API error: {e}"]

def parse_cv_from_text(cv_text: str) -> dict:
    """
    Parses raw CV text using Gemini Pro to extract structured data.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    Based on the following CV text, extract the information and return it as a JSON object.
    The JSON object should have the following keys: "summary", "skills", "education", "experience".
    - "summary": A brief professional summary.
    - "skills": A list of skills. This should be a flat list of strings.
    - "education": A list of educational qualifications, each being an object with "institution", "qualification", and "dates".
    - "experience": A list of work experiences, each being an object with "company", "position", "dates", "responsibilities", and "achievements".

    CV Text:
    ---
    {cv_text}
    ---

    Respond ONLY with the JSON object. Do not include markdown specifiers like ```json.
    """

    try:
        response = model.generate_content(prompt)
        # Clean the response to get a valid JSON string
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"Error parsing CV text with Gemini: {e}")
        raise ValueError("Failed to parse CV with Gemini API.")


def parse_cv_from_image(image_bytes: bytes) -> dict:
    """
    Parses a CV from an image using Gemini Pro Vision to extract structured data.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        cv_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to open image bytes: {e}")
        raise ValueError("Invalid image format.")

    prompt = """
    Based on the CV in the image, extract the information and return it as a JSON object.
    The JSON object should have the following keys: "summary", "skills", "education", "experience".
    - "summary": A brief professional summary.
    - "skills": A list of skills. This should be a flat list of strings.
    - "education": A list of educational qualifications, each being an object with "institution", "qualification", and "dates".
    - "experience": A list of work experiences, each being an object with "company", "position", "dates", "responsibilities", and "achievements".

    Respond ONLY with the JSON object. Do not include markdown specifiers like ```json.
    """

    try:
        response = model.generate_content([prompt, cv_image])
        # Clean the response to get a valid JSON string
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"Error parsing CV image with Gemini Vision: {e}")
        raise ValueError("Failed to parse CV with Gemini API.")

def _get_currency_for_location(location: str) -> str:
    """A simple helper to guess the currency from a location string."""
    location_lower = location.lower()
    # Simple mapping for demonstration purposes. A real app might use a library.
    if any(loc in location_lower for loc in ['uk', 'london', 'united kingdom']):
        return 'GBP'
    if any(loc in location_lower for loc in ['japan', 'tokyo']):
        return 'JPY'
    if any(loc in location_lower for loc in ['germany', 'berlin', 'france', 'paris', 'spain', 'madrid']):
        return 'EUR'
    if any(loc in location_lower for loc in ['canada', 'toronto', 'vancouver']):
        return 'CAD'
    if any(loc in location_lower for loc in ['australia', 'sydney']):
        return 'AUD'
    # Default to USD
    return 'USD'

def _convert_currency(amount: int, from_currency: str, to_currency: str) -> int:
    """Converts an amount from one currency to another using a free API."""
    if from_currency == to_currency:
        return amount
    try:
        response = requests.get(
            f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
        )
        response.raise_for_status()
        data = response.json()
        return int(data['rates'][to_currency])
    except Exception as e:
        logger.error(f"Currency conversion failed: {e}")
        # If conversion fails, return the original amount
        return amount

def get_salary_estimation_with_gemini(
    job_title: str,
    location: str,
    skills: list[str],
    experience: list[dict],
    education: list[dict]
) -> dict:
    """
    Provides a salary estimation using Gemini based on detailed CV data.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Create detailed string representations of the user's profile
    skills_str = ", ".join(skills)
    
    experience_details = []
    for exp in experience:
        experience_details.append(
            f"- {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('period', 'N/A')})"
        )
    experience_str = "\n".join(experience_details)

    education_details = []
    for edu in education:
        education_details.append(
            f"- {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')} ({edu.get('years', 'N/A')})"
        )
    education_str = "\n".join(education_details)

    prompt = f"""
    Act as an expert compensation analyst and career coach.
    Based on the following candidate profile and job details, provide a realistic salary estimation for the role.

    Return your answer as a JSON object with the following keys:
    - "min_salary": The lower end of the likely salary range (integer).
    - "max_salary": The upper end of the likely salary range (integer).
    - "median_salary": The median salary for this profile (integer).
    - "commentary": A brief, insightful commentary (2-3 sentences) explaining the rationale behind the estimate, considering the candidate's experience, skills, and the location.

    IMPORTANT: Provide all salary figures in USD. Do not use any currency symbols.

    Candidate Profile:
    - Location: {location}
    - Skills: {skills_str}
    - Experience:
    {experience_str}
    - Education:
    {education_str}

    Job Title: {job_title}
    
    Respond ONLY with the JSON object. Do not include markdown specifiers like ```json.
    """

    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
        usd_salary_data = json.loads(json_string)

        # --- Currency Conversion Step ---
        target_currency = _get_currency_for_location(location)
        if target_currency != 'USD':
            usd_salary_data['min_salary'] = _convert_currency(usd_salary_data['min_salary'], 'USD', target_currency)
            usd_salary_data['max_salary'] = _convert_currency(usd_salary_data['max_salary'], 'USD', target_currency)
            usd_salary_data['median_salary'] = _convert_currency(usd_salary_data['median_salary'], 'USD', target_currency)

        # Add currency and period to the response
        usd_salary_data['currency'] = target_currency
        usd_salary_data['period'] = 'annual'

        return usd_salary_data

    except Exception as e:
        logger.error(f"Error getting salary estimation from Gemini: {e}")
        raise ValueError("Failed to get salary estimation from Gemini API.") 