from typing import Dict, Any

def analyze_answer_with_star(answer: str) -> str:
    """
    Analyzes a user's answer to an interview question using the STAR method.
    This is a placeholder and will be replaced with a more sophisticated NLP model.
    """
    # Placeholder logic
    feedback = "Thank you for your answer. "
    if "situation" in answer.lower() or "task" in answer.lower():
        feedback += "You've described the situation well. "
    if "action" in answer.lower():
        feedback += "You've clearly outlined the action you took. "
    if "result" in answer.lower():
        feedback += "It's great that you mentioned the result. "
    
    if len(feedback) < 50:
        return "That's a good start. Can you tell me more about the situation, the task you were assigned, the action you took, and the result of that action?"
    
    return feedback

def generate_questions_from_cv(cv_data: Dict[str, Any], job_description: str) -> list[str]:
    """
    Generates interview questions based on the user's CV and the job description.
    This will use a fine-tuned model.
    """
    # Placeholder for the fine-tuned model logic
    skills = ", ".join(cv_data.get('skills', []))
    return [
        f"Based on your experience at {cv_data.get('experiences', [{}])[0].get('company', 'your last company')}, can you describe a challenging project you worked on?",
        f"How have you used your skills in {skills} to solve a complex problem?",
        "What interests you most about this role?",
        "Describe a time you had to collaborate with a difficult team member.",
        "Where do you see yourself in five years?"
    ] 