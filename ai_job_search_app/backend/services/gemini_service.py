import os
import google.generativeai as genai
import ast

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # This check is important for startup.
    # If the key is missing, the server will fail to start with a clear error.
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
# --- End Configuration ---


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
        
        # The response text is a multi-line string.
        response_text = response.text.strip()
        
        # Split the text by newlines and filter out any empty lines
        questions = [q.strip() for q in response_text.split('\n') if q.strip()]
        
        # Basic validation
        if not questions:
            return ["Error: Model returned an empty response."]
            
        return questions

    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        # In case of any error (API, parsing, etc.), return a helpful error message.
        return [f"Failed to generate questions due to an API error: {e}"]

def parse_cv_with_gemini(cv_text: str) -> dict:
    """
    Uses the Gemini API to parse a CV and extract structured information.

    Args:
        cv_text: The full text of the user's CV.

    Returns:
        A dictionary containing the parsed CV data.
    """
    prompt = f"""
    Act as an expert CV parser and data extractor. Your task is to analyze the provided CV text and extract the following information in a structured JSON format:
    - summary: A brief 2-3 sentence professional summary of the candidate.
    - skills: A list of key technical and soft skills.
    - educations: A list of educational qualifications, including 'institution', 'degree', and 'years'.
    - experiences: A list of professional experiences, including 'company', 'role', 'period', and a brief 'description' of responsibilities.

    Please provide the output as a single JSON object. Do not include any explanatory text before or after the JSON.

    Example format:
    {{
        "summary": "A highly motivated software engineer with 5 years of experience...",
        "skills": ["Python", "FastAPI", "React", "AWS"],
        "educations": [
            {{"institution": "University of Technology", "degree": "B.S. in Computer Science", "years": "2015-2019"}}
        ],
        "experiences": [
            {{
                "company": "Tech Solutions Inc.",
                "role": "Software Engineer",
                "period": "2019-Present",
                "description": "Developed and maintained web applications using Python and FastAPI."
            }}
        ]
    }}

    ---
    CV TEXT:
    {cv_text}
    ---
    PARSED JSON OUTPUT:
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # The model should return a JSON string. We'll clean it up and parse it.
        # It often includes ```json ... ``` which we need to remove.
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        response_text = response_text.strip()
        
        # Safely evaluate the string to a Python dict
        parsed_data = ast.literal_eval(response_text)
        
        return parsed_data

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response was: {response.text}")
        raise ValueError("Failed to parse CV data from the model's response.")
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        raise IOError("An error occurred while communicating with the Gemini API.") 