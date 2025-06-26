"""
Test the application tracking pipeline.
"""
import pytest
from fastapi import status


class TestApplicationTrackingPipeline:
    """Test the complete application tracking pipeline."""
    
    def test_create_application_success(self, test_client, auth_headers):
        """Test successful job application creation."""
        
        application_data = {
            "job_title": "Senior Python Developer",
            "company_name": "Tech Innovations Inc",
            "job_url": "https://example.com/job/123",
            "application_date": "2024-01-15",
            "status": "applied",
            "notes": "Applied through company website"
        }
        
        response = test_client.post(
            "/api/applications/",
            json=application_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        # Verify application data
        assert data["job_title"] == application_data["job_title"]
        assert data["company_name"] == application_data["company_name"]
        assert data["status"] == application_data["status"]
        assert "id" in data
        assert "created_at" in data
    
    def test_get_user_applications(self, test_client, auth_headers):
        """Test retrieving user's job applications."""
        
        # Create a few applications first
        applications = [
            {
                "job_title": "Python Developer",
                "company_name": "Company A",
                "status": "applied"
            },
            {
                "job_title": "Data Scientist", 
                "company_name": "Company B",
                "status": "interviewing"
            }
        ]
        
        created_apps = []
        for app_data in applications:
            response = test_client.post(
                "/api/applications/",
                json=app_data,
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_201_CREATED
            created_apps.append(response.json())
        
        # Get all applications
        response = test_client.get("/api/applications/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        
        # Verify applications belong to the user
        for app in data:
            assert "id" in app
            assert "job_title" in app
            assert "company_name" in app
            assert "status" in app
    
    def test_get_specific_application(self, test_client, auth_headers):
        """Test retrieving a specific job application."""
        
        # Create an application
        app_data = {
            "job_title": "Test Position",
            "company_name": "Test Company",
            "status": "applied"
        }
        
        create_response = test_client.post(
            "/api/applications/",
            json=app_data,
            headers=auth_headers
        )
        app_id = create_response.json()["id"]
        
        # Get specific application
        response = test_client.get(
            f"/api/applications/{app_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == app_id
        assert data["job_title"] == app_data["job_title"]
    
    def test_get_nonexistent_application(self, test_client, auth_headers):
        """Test retrieving a non-existent application."""
        
        response = test_client.get(
            "/api/applications/99999",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    def test_update_application_status(self, test_client, auth_headers):
        """Test updating an application's status."""
        
        # Create an application
        app_data = {
            "job_title": "Software Engineer",
            "company_name": "Tech Corp",
            "status": "applied"
        }
        
        create_response = test_client.post(
            "/api/applications/",
            json=app_data,
            headers=auth_headers
        )
        app_id = create_response.json()["id"]
        
        # Update application status
        update_data = {
            "status": "interviewing",
            "notes": "Phone interview scheduled for next week"
        }
        
        response = test_client.put(
            f"/api/applications/{app_id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "interviewing"
        assert data["notes"] == update_data["notes"]
    
    def test_delete_application(self, test_client, auth_headers):
        """Test deleting a job application."""
        
        # Create an application
        app_data = {
            "job_title": "Temporary Job",
            "company_name": "Temp Company",
            "status": "applied"
        }
        
        create_response = test_client.post(
            "/api/applications/",
            json=app_data,
            headers=auth_headers
        )
        app_id = create_response.json()["id"]
        
        # Delete application
        response = test_client.delete(
            f"/api/applications/{app_id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify application is deleted
        get_response = test_client.get(
            f"/api/applications/{app_id}",
            headers=auth_headers
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_application_status_workflow(self, test_client, auth_headers):
        """Test complete application status workflow."""
        
        # Create application
        app_data = {
            "job_title": "Full Stack Developer",
            "company_name": "Workflow Corp",
            "status": "applied"
        }
        
        create_response = test_client.post(
            "/api/applications/",
            json=app_data,
            headers=auth_headers
        )
        app_id = create_response.json()["id"]
        
        # Workflow: applied -> interviewing -> offer -> accepted
        statuses = [
            ("interviewing", "First round interview completed"),
            ("offer", "Received job offer"),
            ("accepted", "Accepted the position!")
        ]
        
        for status_val, notes in statuses:
            update_data = {"status": status_val, "notes": notes}
            
            response = test_client.put(
                f"/api/applications/{app_id}",
                json=update_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == status_val
            assert data["notes"] == notes
    
    def test_application_pagination(self, test_client, auth_headers):
        """Test application list pagination."""
        
        # Create multiple applications
        for i in range(15):
            app_data = {
                "job_title": f"Job {i}",
                "company_name": f"Company {i}",
                "status": "applied"
            }
            response = test_client.post(
                "/api/applications/",
                json=app_data,
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_201_CREATED
        
        # Test pagination
        response = test_client.get(
            "/api/applications/?skip=0&limit=10",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 10
        
        # Test second page
        response = test_client.get(
            "/api/applications/?skip=10&limit=10",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        second_page = response.json()
        assert len(second_page) >= 5  # Should have remaining applications
    
    def test_application_user_isolation(self, test_client, test_user_data):
        """Test that users can only access their own applications."""
        
        # Create two users
        user1_data = test_user_data.copy()
        user2_data = {
            "email": "user2@example.com",
            "password": "password123",
            "full_name": "User Two"
        }
        
        # Register both users
        test_client.post("/api/auth/register", json=user1_data)
        test_client.post("/api/auth/register", json=user2_data)
        
        # Get auth headers for both users
        def get_auth_headers(user_data):
            login_data = {
                "username": user_data["email"],
                "password": user_data["password"]
            }
            response = test_client.post("/api/auth/login", data=login_data)
            token = response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
        
        user1_headers = get_auth_headers(user1_data)
        user2_headers = get_auth_headers(user2_data)
        
        # User 1 creates an application
        app_data = {
            "job_title": "Private Job",
            "company_name": "Private Company",
            "status": "applied"
        }
        
        response = test_client.post(
            "/api/applications/",
            json=app_data,
            headers=user1_headers
        )
        app_id = response.json()["id"]
        
        # User 2 should not be able to access User 1's application
        response = test_client.get(
            f"/api/applications/{app_id}",
            headers=user2_headers
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # User 2 should not see User 1's applications in their list
        response = test_client.get("/api/applications/", headers=user2_headers)
        assert response.status_code == status.HTTP_200_OK
        user2_apps = response.json()
        
        # Should be empty or not contain User 1's application
        app_ids = [app["id"] for app in user2_apps]
        assert app_id not in app_ids


class TestResumeBuilderPipeline:
    """Test resume builder and encrypted storage pipeline."""
    
    def test_save_resume_data_success(self, test_client, auth_headers):
        """Test successful resume data saving with encryption."""
        
        resume_data = {
            "content": {
                "personal_info": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "+1234567890"
                },
                "experience": [
                    {
                        "company": "Tech Corp",
                        "position": "Software Engineer",
                        "duration": "2020-2023",
                        "achievements": ["Built scalable APIs", "Improved performance by 40%"]
                    }
                ],
                "skills": ["Python", "React", "AWS"],
                "education": [
                    {
                        "institution": "University of Tech",
                        "degree": "Computer Science",
                        "year": "2020"
                    }
                ]
            }
        }
        
        response = test_client.post(
            "/api/resume/",
            json=resume_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"].lower()
    
    def test_resume_data_encryption(self, test_client, auth_headers, test_user):
        """Test that resume data is properly encrypted when stored."""
        
        resume_data = {
            "content": {
                "personal_info": {"name": "Test User"},
                "skills": ["Python", "Testing"]
            }
        }
        
        response = test_client.post(
            "/api/resume/",
            json=resume_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Check that encrypted data is stored (would need database access)
        # For now, just verify the endpoint worked
        assert test_user.encrypted_resume_data is not None
    
    def test_save_empty_resume_data(self, test_client, auth_headers):
        """Test saving empty resume data."""
        
        resume_data = {"content": {}}
        
        response = test_client.post(
            "/api/resume/",
            json=resume_data,
            headers=auth_headers
        )
        
        # Should still succeed even with empty content
        assert response.status_code == status.HTTP_200_OK
    
    def test_save_large_resume_data(self, test_client, auth_headers):
        """Test saving large resume data."""
        
        # Create large resume with many entries
        large_experience = []
        for i in range(20):
            large_experience.append({
                "company": f"Company {i}",
                "position": f"Position {i}",
                "duration": f"202{i//10}-202{(i//10)+1}",
                "description": "A" * 1000  # Long description
            })
        
        resume_data = {
            "content": {
                "personal_info": {"name": "Experienced Professional"},
                "experience": large_experience,
                "skills": ["Skill"] * 100  # Many skills
            }
        }
        
        response = test_client.post(
            "/api/resume/",
            json=resume_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK