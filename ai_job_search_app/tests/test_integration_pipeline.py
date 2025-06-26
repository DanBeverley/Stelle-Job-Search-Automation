"""
Test the complete end-to-end integration pipeline.
"""
import pytest
from unittest.mock import patch
from fastapi import status
from io import BytesIO


class TestCompleteUserJourney:
    """Test the complete user journey through the application."""
    
    def test_complete_user_workflow(self, test_client, test_user_data, mock_ml_models, mock_external_apis):
        """Test complete user workflow from registration to job search and AI features."""
        
        # Step 1: User Registration
        register_response = test_client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == status.HTTP_200_OK
        
        # Step 2: User Login
        login_data = {
            "username": test_user_data["email"],
            "password": test_user_data["password"]
        }
        login_response = test_client.post("/api/auth/login", data=login_data)
        assert login_response.status_code == status.HTTP_200_OK
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 3: CV Upload and Parsing
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_extract_skills:
            
            mock_extract_pdf.return_value = "John Doe Software Engineer Python ML"
            mock_extract_skills.return_value = ["Python", "Machine Learning", "FastAPI"]
            
            pdf_content = b"Sample CV content"
            files = {"file": ("resume.pdf", BytesIO(pdf_content), "application/pdf")}
            
            cv_response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=headers
            )
            assert cv_response.status_code == status.HTTP_200_OK
        
        # Step 4: Verify CV Data
        cv_status_response = test_client.get("/api/cv/parse-status", headers=headers)
        assert cv_status_response.status_code == status.HTTP_200_OK
        cv_data = cv_status_response.json()
        assert "skills" in cv_data
        
        # Step 5: Job Search
        job_search_response = test_client.get(
            "/api/jobs/search",
            params={"keyword": "Python Developer", "location": "San Francisco"},
            headers=headers
        )
        assert job_search_response.status_code == status.HTTP_200_OK
        jobs = job_search_response.json()["jobs"]
        assert len(jobs) > 0
        
        # Step 6: Salary Prediction
        with patch('backend.services.salary_prediction_service.predict_salary_with_xgb') as mock_salary:
            mock_salary.return_value = {
                "min_salary": 90000,
                "max_salary": 130000,
                "median_salary": 110000,
                "commentary": "Based on your experience",
                "currency": "USD",
                "period": "annual"
            }
            
            salary_response = test_client.post(
                "/api/salary/predict",
                json={
                    "job_title": "Senior Python Developer",
                    "location": "San Francisco",
                    "skills": ["Python", "Machine Learning"]
                },
                headers=headers
            )
            assert salary_response.status_code == status.HTTP_200_OK
            salary_data = salary_response.json()
            assert salary_data["median_salary"] > 0
        
        # Step 7: Skill Analysis
        with patch('backend.services.skill_analysis_service.analyze_skill_gap') as mock_skills:
            mock_skills.return_value = {
                "matched_skills": ["Python"],
                "missing_skills": ["React", "TypeScript"],
                "recommended_courses": [
                    {"title": "React Course", "provider": "edX", "url": "example.com"}
                ]
            }
            
            skills_response = test_client.post(
                "/api/skills/skill-analysis",
                json={"job_description": "Python and React developer needed"},
                headers=headers
            )
            assert skills_response.status_code == status.HTTP_200_OK
            skills_data = skills_response.json()
            assert "matched_skills" in skills_data
            assert "missing_skills" in skills_data
        
        # Step 8: Cover Letter Generation
        with patch('backend.services.cover_letter_service.generate_cover_letter_with_finetuned_model') as mock_cover:
            mock_cover.return_value = {
                "cover_letter": "Dear Hiring Manager, I am excited to apply for the Python Developer position..."
            }
            
            cover_response = test_client.post(
                "/api/cover-letter/generate",
                json={
                    "job_title": "Python Developer",
                    "company_name": "Tech Corp",
                    "job_description": "We need a Python developer"
                },
                headers=headers
            )
            assert cover_response.status_code == status.HTTP_200_OK
            cover_data = cover_response.json()
            assert len(cover_data["cover_letter"]) > 50
        
        # Step 9: Interview Preparation
        with patch('backend.services.interview_prep_service.generate_questions_from_cv') as mock_interview:
            mock_interview.return_value = [
                "Tell me about your Python experience",
                "How have you used Machine Learning in projects?"
            ]
            
            interview_response = test_client.post(
                "/api/interview/interview-prep/generate-questions",
                json={"job_description": "Python developer with ML experience"},
                headers=headers
            )
            assert interview_response.status_code == status.HTTP_200_OK
            interview_data = interview_response.json()
            assert len(interview_data["questions"]) > 0
        
        # Step 10: Create Job Application
        application_response = test_client.post(
            "/api/applications/",
            json={
                "job_title": "Python Developer",
                "company_name": "Tech Corp",
                "status": "applied",
                "job_url": "https://example.com/job",
                "notes": "Applied after using AI features"
            },
            headers=headers
        )
        assert application_response.status_code == status.HTTP_201_CREATED
        
        # Step 11: Save Resume
        resume_response = test_client.post(
            "/api/resume/",
            json={
                "content": {
                    "personal_info": {"name": "John Doe"},
                    "skills": ["Python", "Machine Learning"],
                    "experience": [{"company": "Previous Corp", "position": "Engineer"}]
                }
            },
            headers=headers
        )
        assert resume_response.status_code == status.HTTP_200_OK
        
        # Verify complete workflow succeeded
        print("✅ Complete user workflow test passed!")
    
    def test_error_recovery_workflow(self, test_client, test_user_data, mock_ml_models):
        """Test error recovery in the workflow when services fail."""
        
        # Register and login user
        test_client.post("/api/auth/register", json=test_user_data)
        login_data = {
            "username": test_user_data["email"],
            "password": test_user_data["password"]
        }
        login_response = test_client.post("/api/auth/login", data=login_data)
        headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}
        
        # Test CV upload failure recovery
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract:
            mock_extract.side_effect = Exception("File corrupted")
            
            pdf_content = b"Corrupted PDF"
            files = {"file": ("bad_resume.pdf", BytesIO(pdf_content), "application/pdf")}
            
            cv_response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=headers
            )
            # Should handle error gracefully
            assert cv_response.status_code in [400, 500]
        
        # Upload working CV for remaining tests
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_skills:
            
            mock_extract.return_value = "Good CV content"
            mock_skills.return_value = ["Python"]
            
            pdf_content = b"Good PDF"
            files = {"file": ("good_resume.pdf", BytesIO(pdf_content), "application/pdf")}
            
            cv_response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=headers
            )
            assert cv_response.status_code == status.HTTP_200_OK
        
        # Test ML service failure recovery
        with patch('backend.services.salary_prediction_service.predict_salary_with_xgb') as mock_salary:
            mock_salary.side_effect = RuntimeError("Model not available")
            
            salary_response = test_client.post(
                "/api/salary/predict",
                json={"job_title": "Developer", "location": "NYC", "skills": ["Python"]},
                headers=headers
            )
            assert salary_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        
        # Test external API failure recovery
        with patch('backend.services.job_search_providers.adzuna_api.search_adzuna_jobs') as mock_adzuna, \
             patch('backend.services.job_search_providers.theirstack_api.search_theirstack_jobs') as mock_theirstack:
            
            mock_adzuna.side_effect = Exception("Adzuna down")
            mock_theirstack.side_effect = Exception("TheirStack down")
            
            job_response = test_client.get(
                "/api/jobs/search",
                params={"keyword": "Developer", "location": "NYC"},
                headers=headers
            )
            # Should fail when both providers are down
            assert job_response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_data_consistency_across_features(self, test_client, auth_headers, test_user_with_cv, mock_ml_models):
        """Test that CV data is consistently used across all AI features."""
        
        # Mock all services to track CV data usage
        with patch('backend.services.salary_prediction_service.predict_salary_with_xgb') as mock_salary, \
             patch('backend.services.skill_analysis_service.analyze_skill_gap') as mock_skills, \
             patch('backend.services.cover_letter_service.generate_cover_letter_with_finetuned_model') as mock_cover, \
             patch('backend.services.interview_prep_service.generate_questions_from_cv') as mock_interview:
            
            # Configure mocks to return consistent data
            mock_salary.return_value = {"min_salary": 80000, "max_salary": 120000, "median_salary": 100000, "commentary": "Test", "currency": "USD", "period": "annual"}
            mock_skills.return_value = {"matched_skills": ["Python"], "missing_skills": ["React"], "recommended_courses": []}
            mock_cover.return_value = {"cover_letter": "Test cover letter"}
            mock_interview.return_value = ["Test question"]
            
            # Call all AI features
            job_description = "Python developer with ML experience needed"
            
            # Salary prediction
            test_client.post(
                "/api/salary/predict",
                json={"job_title": "Developer", "location": "NYC", "skills": ["Python"]},
                headers=auth_headers
            )
            
            # Skill analysis
            test_client.post(
                "/api/skills/skill-analysis",
                json={"job_description": job_description},
                headers=auth_headers
            )
            
            # Cover letter
            test_client.post(
                "/api/cover-letter/generate",
                json={"job_title": "Developer", "company_name": "Corp", "job_description": job_description},
                headers=auth_headers
            )
            
            # Interview prep
            test_client.post(
                "/api/interview/interview-prep/generate-questions",
                json={"job_description": job_description},
                headers=auth_headers
            )
            
            # All services should have been called
            assert mock_salary.called
            assert mock_skills.called
            assert mock_cover.called
            assert mock_interview.called
    
    def test_concurrent_requests_handling(self, test_client, auth_headers, test_user_with_cv, mock_ml_models):
        """Test handling of concurrent requests to AI services."""
        import threading
        import time
        
        results = []
        
        def make_request(endpoint, data):
            try:
                response = test_client.post(endpoint, json=data, headers=auth_headers)
                results.append(("success", response.status_code))
            except Exception as e:
                results.append(("error", str(e)))
        
        # Mock services
        with patch('backend.services.salary_prediction_service.predict_salary_with_xgb') as mock_salary, \
             patch('backend.services.skill_analysis_service.analyze_skill_gap') as mock_skills:
            
            mock_salary.return_value = {"min_salary": 80000, "max_salary": 120000, "median_salary": 100000, "commentary": "Test", "currency": "USD", "period": "annual"}
            mock_skills.return_value = {"matched_skills": ["Python"], "missing_skills": [], "recommended_courses": []}
            
            # Create concurrent threads
            threads = []
            
            # Salary prediction requests
            for i in range(3):
                thread = threading.Thread(
                    target=make_request,
                    args=(
                        "/api/salary/predict",
                        {"job_title": f"Developer {i}", "location": "NYC", "skills": ["Python"]}
                    )
                )
                threads.append(thread)
            
            # Skill analysis requests
            for i in range(3):
                thread = threading.Thread(
                    target=make_request,
                    args=(
                        "/api/skills/skill-analysis",
                        {"job_description": f"Job description {i}"}
                    )
                )
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)
            
            # Check results
            assert len(results) == 6
            successful_requests = [r for r in results if r[0] == "success" and r[1] == 200]
            assert len(successful_requests) >= 4  # Most should succeed


class TestSystemHealthAndMonitoring:
    """Test system health and monitoring capabilities."""
    
    def test_api_health_check(self, test_client):
        """Test basic API health check."""
        response = test_client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "AI Job Search API" in data["message"]
    
    def test_database_connectivity(self, test_client, test_user_data):
        """Test database connectivity through user operations."""
        # Test database write
        response = test_client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == status.HTTP_200_OK
        
        # Test database read
        login_data = {
            "username": test_user_data["email"],
            "password": test_user_data["password"]
        }
        response = test_client.post("/api/auth/login", data=login_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_ml_models_loading_status(self, test_client, auth_headers, mock_ml_models):
        """Test ML models loading and availability."""
        # This would test actual model loading in a real environment
        # For now, test that mocked models respond correctly
        
        with patch('backend.services.salary_prediction_service.predict_salary_with_xgb') as mock_salary:
            mock_salary.return_value = {"min_salary": 80000, "max_salary": 120000, "median_salary": 100000, "commentary": "Test", "currency": "USD", "period": "annual"}
            
            # Test salary model
            response = test_client.post(
                "/api/salary/predict",
                json={"job_title": "Test", "location": "Test", "skills": ["Test"]},
                headers=auth_headers
            )
            
            # Should not be 503 (service unavailable) if models are loaded
            assert response.status_code != status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_external_api_connectivity(self, test_client, auth_headers, mock_external_apis):
        """Test external API connectivity."""
        response = test_client.get(
            "/api/jobs/search",
            params={"keyword": "Test", "location": "Test City"},
            headers=auth_headers
        )
        
        # Should successfully call external APIs
        assert response.status_code == status.HTTP_200_OK
        mock_external_apis["adzuna"].assert_called()
        mock_external_apis["theirstack"].assert_called()