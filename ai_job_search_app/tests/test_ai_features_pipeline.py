"""
Test the AI-powered features pipeline (salary prediction, skill analysis, cover letter, interview prep).
"""
import pytest
from unittest.mock import patch
from fastapi import status


class TestSalaryPredictionPipeline:
    """Test salary prediction feature pipeline."""
    
    def test_salary_prediction_success(self, test_client, auth_headers, test_user_with_cv, mock_ml_models):
        """Test successful salary prediction."""
        
        request_data = {
            "job_title": "Senior Python Developer",
            "location": "San Francisco, CA",
            "skills": ["Python", "Machine Learning", "FastAPI"]
        }
        
        response = test_client.post(
            "/api/salary/predict",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify salary prediction response structure
        required_fields = ["min_salary", "max_salary", "median_salary", "commentary", "currency", "period"]
        for field in required_fields:
            assert field in data
        
        # Verify reasonable salary values
        assert data["min_salary"] > 0
        assert data["max_salary"] > data["min_salary"]
        assert data["median_salary"] >= data["min_salary"]
        assert data["median_salary"] <= data["max_salary"]
        assert data["currency"] == "USD"
        assert data["period"] == "annual"
    
    def test_salary_prediction_no_cv_data(self, test_client, auth_headers):
        """Test salary prediction when user has no CV data."""
        
        request_data = {
            "job_title": "Developer",
            "location": "NYC",
            "skills": ["Python"]
        }
        
        response = test_client.post(
            "/api/salary/predict",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "cv data not found" in response.json()["detail"].lower()
    
    def test_salary_prediction_model_failure(self, test_client, auth_headers, test_user_with_cv, mock_ml_models):
        """Test salary prediction when ML model fails."""
        
        # Mock model failure
        mock_ml_models["salary_predictor"].predict.side_effect = RuntimeError("Model unavailable")
        
        request_data = {
            "job_title": "Engineer",
            "location": "Boston",
            "skills": ["Java"]
        }
        
        response = test_client.post(
            "/api/salary/predict",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestSkillAnalysisPipeline:
    """Test skill analysis feature pipeline."""
    
    def test_skill_analysis_success(self, test_client, auth_headers, test_user_with_cv):
        """Test successful skill gap analysis."""
        
        with patch('backend.services.skill_analysis_service.analyze_skill_gap') as mock_analysis:
            
            mock_analysis.return_value = {
                "matched_skills": ["Python", "Machine Learning"],
                "missing_skills": ["React", "Node.js"],
                "skill_categories": {
                    "programming": ["Python"],
                    "frameworks": ["React", "Node.js"],
                    "ml": ["Machine Learning"]
                },
                "recommended_courses": [
                    {
                        "title": "React Fundamentals",
                        "provider": "edX",
                        "url": "https://example.com/react-course"
                    }
                ]
            }
            
            request_data = {
                "job_description": "We need a developer with Python, React, and Node.js experience"
            }
            
            response = test_client.post(
                "/api/skills/skill-analysis",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "matched_skills" in data
            assert "missing_skills" in data
            assert "recommended_courses" in data
            assert len(data["matched_skills"]) > 0
            assert len(data["missing_skills"]) > 0
    
    def test_skill_analysis_no_skills_in_cv(self, test_client, auth_headers, test_user_with_cv):
        """Test skill analysis when CV has no skills."""
        
        # Remove skills from CV data
        test_user_with_cv.parsed_cv_data["skills"] = []
        
        request_data = {
            "job_description": "Python developer needed"
        }
        
        response = test_client.post(
            "/api/skills/skill-analysis",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "no skills found" in response.json()["detail"].lower()


class TestCoverLetterPipeline:
    """Test cover letter generation pipeline."""
    
    def test_cover_letter_generation_success(self, test_client, auth_headers, mock_ml_models):
        """Test successful cover letter generation."""
        
        with patch('backend.services.cover_letter_service.generate_cover_letter_with_finetuned_model') as mock_generate:
            
            mock_generate.return_value = {
                "cover_letter": "Dear Hiring Manager,\n\nI am writing to express my interest...",
                "length": 250,
                "tone": "professional"
            }
            
            request_data = {
                "job_title": "Software Engineer",
                "company_name": "Tech Corp",
                "job_description": "We are looking for a Python developer..."
            }
            
            response = test_client.post(
                "/api/cover-letter/generate",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "cover_letter" in data
            assert len(data["cover_letter"]) > 0
            assert "Dear Hiring Manager" in data["cover_letter"]
    
    def test_cover_letter_generation_model_failure(self, test_client, auth_headers, mock_ml_models):
        """Test cover letter generation when model fails."""
        
        with patch('backend.services.cover_letter_service.generate_cover_letter_with_finetuned_model') as mock_generate:
            
            mock_generate.side_effect = RuntimeError("Model not loaded")
            
            request_data = {
                "job_title": "Developer",
                "company_name": "Company",
                "job_description": "Job description"
            }
            
            response = test_client.post(
                "/api/cover-letter/generate",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestInterviewPrepPipeline:
    """Test interview preparation pipeline."""
    
    def test_interview_questions_generation_success(self, test_client, auth_headers, test_user_with_cv, mock_ml_models):
        """Test successful interview questions generation."""
        
        with patch('backend.services.interview_prep_service.generate_questions_from_cv') as mock_questions:
            
            mock_questions.return_value = [
                "Tell me about your experience with Python.",
                "How have you used Machine Learning in your projects?",
                "Describe a challenging problem you solved with FastAPI."
            ]
            
            request_data = {
                "job_description": "We need a Python developer with ML experience"
            }
            
            response = test_client.post(
                "/api/interview/interview-prep/generate-questions",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "questions" in data
            assert len(data["questions"]) > 0
            assert all(isinstance(q, str) for q in data["questions"])
    
    def test_interview_questions_empty_job_description(self, test_client, auth_headers, test_user_with_cv):
        """Test interview questions with empty job description."""
        
        request_data = {"job_description": ""}
        
        response = test_client.post(
            "/api/interview/interview-prep/generate-questions",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot be empty" in response.json()["detail"].lower()
    
    def test_star_method_answer_analysis(self, test_client, auth_headers):
        """Test STAR method answer analysis."""
        
        with patch('backend.services.interview_prep_service.analyze_answer_with_star') as mock_analysis:
            
            mock_analysis.return_value = {
                "situation": "Present but could be more specific",
                "task": "Clearly defined", 
                "action": "Well detailed",
                "result": "Quantified results provided",
                "overall_score": 8.5,
                "feedback": "Good use of STAR method with room for improvement in situation description"
            }
            
            request_data = {
                "answer": "In my previous role as a software engineer at Tech Corp, I was tasked with improving the API response time. I analyzed the bottlenecks, implemented caching, and reduced response time by 40%."
            }
            
            response = test_client.post(
                "/api/interview/interview-prep/analyze-answer",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "feedback" in data
            assert isinstance(data["feedback"], dict)
    
    def test_star_analysis_empty_answer(self, test_client, auth_headers):
        """Test STAR analysis with empty answer."""
        
        request_data = {"answer": ""}
        
        response = test_client.post(
            "/api/interview/interview-prep/analyze-answer",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot be empty" in response.json()["detail"].lower()


class TestAIFeaturesIntegration:
    """Test integration between AI features."""
    
    def test_ai_features_with_same_cv_data(self, test_client, auth_headers, test_user_with_cv, mock_ml_models):
        """Test that all AI features work with the same CV data consistently."""
        
        # Mock all AI services
        with patch('backend.services.salary_prediction_service.predict_salary_with_xgb') as mock_salary, \
             patch('backend.services.skill_analysis_service.analyze_skill_gap') as mock_skills, \
             patch('backend.services.cover_letter_service.generate_cover_letter_with_finetuned_model') as mock_cover, \
             patch('backend.services.interview_prep_service.generate_questions_from_cv') as mock_interview:
            
            # Configure mocks
            mock_salary.return_value = {
                "min_salary": 80000,
                "max_salary": 120000,
                "median_salary": 100000,
                "commentary": "Based on your skills",
                "currency": "USD",
                "period": "annual"
            }
            
            mock_skills.return_value = {
                "matched_skills": ["Python"],
                "missing_skills": ["React"],
                "recommended_courses": []
            }
            
            mock_cover.return_value = {
                "cover_letter": "Professional cover letter content"
            }
            
            mock_interview.return_value = [
                "Tell me about your Python experience"
            ]
            
            # Test salary prediction
            salary_response = test_client.post(
                "/api/salary/predict",
                json={"job_title": "Developer", "location": "NYC", "skills": ["Python"]},
                headers=auth_headers
            )
            assert salary_response.status_code == status.HTTP_200_OK
            
            # Test skill analysis  
            skills_response = test_client.post(
                "/api/skills/skill-analysis",
                json={"job_description": "Python developer needed"},
                headers=auth_headers
            )
            assert skills_response.status_code == status.HTTP_200_OK
            
            # Test cover letter
            cover_response = test_client.post(
                "/api/cover-letter/generate",
                json={"job_title": "Developer", "company_name": "Corp", "job_description": "Python role"},
                headers=auth_headers
            )
            assert cover_response.status_code == status.HTTP_200_OK
            
            # Test interview prep
            interview_response = test_client.post(
                "/api/interview/interview-prep/generate-questions",
                json={"job_description": "Python developer needed"},
                headers=auth_headers
            )
            assert interview_response.status_code == status.HTTP_200_OK
            
            # All features should have accessed the same CV data
            assert all(mock.called for mock in [mock_salary, mock_skills, mock_cover, mock_interview])