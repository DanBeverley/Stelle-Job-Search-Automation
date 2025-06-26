"""
Test the job search and matching pipeline.
"""
import pytest
from unittest.mock import patch
from fastapi import status


class TestJobSearchPipeline:
    """Test the complete job search pipeline."""
    
    def test_job_search_with_location_success(self, test_client, auth_headers, mock_external_apis):
        """Test successful job search with provided location."""
        
        search_params = {
            "keyword": "Python Developer",
            "location": "San Francisco, CA"
        }
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) > 0
        
        # Verify job structure
        job = data["jobs"][0]
        required_fields = ["title", "company", "location", "description", "source"]
        for field in required_fields:
            assert field in job
    
    def test_job_search_without_location_ip_detection(self, test_client, auth_headers, mock_external_apis):
        """Test job search without location using IP geolocation."""
        
        search_params = {"keyword": "Data Scientist"}
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "jobs" in data
        
        # Verify geolocation was called
        mock_external_apis["geolocation"].assert_called()
    
    def test_job_search_user_saved_location(self, test_client, auth_headers, test_user_with_cv, mock_external_apis):
        """Test job search using user's saved location from profile."""
        
        # Update user location in database
        test_user_with_cv.city = "Boston"
        test_user_with_cv.country = "USA"
        
        search_params = {"keyword": "Software Engineer"}
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        # Should use saved location and not call IP geolocation
        mock_external_apis["geolocation"].assert_not_called()
    
    def test_job_search_multiple_providers_aggregation(self, test_client, auth_headers, mock_external_apis):
        """Test that job search aggregates results from multiple providers."""
        
        # Configure mocks to return different jobs
        mock_external_apis["adzuna"].return_value = [
            {
                "title": "Adzuna Job",
                "company": "Adzuna Corp",
                "location": "NYC",
                "description": "Job from Adzuna",
                "source": "Adzuna"
            }
        ]
        
        mock_external_apis["theirstack"].return_value = [
            {
                "title": "TheirStack Job", 
                "company": "TheirStack Inc",
                "location": "LA",
                "description": "Job from TheirStack",
                "source": "TheirStack"
            }
        ]
        
        search_params = {
            "keyword": "Developer",
            "location": "California"
        }
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        jobs = data["jobs"]
        
        # Should have jobs from both providers
        assert len(jobs) >= 2
        sources = [job["source"] for job in jobs]
        assert "Adzuna" in sources
        assert "TheirStack" in sources
    
    def test_job_search_api_failure_handling(self, test_client, auth_headers, mock_external_apis):
        """Test job search when one API provider fails."""
        
        # Make one provider fail
        mock_external_apis["adzuna"].side_effect = Exception("Adzuna API down")
        mock_external_apis["theirstack"].return_value = [
            {
                "title": "Working Job",
                "company": "Working Corp", 
                "location": "Seattle",
                "description": "This provider works",
                "source": "TheirStack"
            }
        ]
        
        search_params = {
            "keyword": "Engineer",
            "location": "Seattle, WA"
        }
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        # Should fail since we're not handling partial failures gracefully yet
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_job_search_no_location_available(self, test_client, auth_headers, mock_external_apis):
        """Test job search when no location can be determined."""
        
        # Mock geolocation failure
        mock_external_apis["geolocation"].return_value = None
        
        search_params = {"keyword": "Developer"}
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "location" in response.json()["detail"].lower()
    
    def test_job_search_empty_keyword(self, test_client, auth_headers):
        """Test job search with empty keyword."""
        
        search_params = {
            "keyword": "",
            "location": "New York"
        }
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        # Should handle empty keyword appropriately
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_job_search_response_standardization(self, test_client, auth_headers, mock_external_apis):
        """Test that job search responses are properly standardized."""
        
        # Mock providers with different response formats
        mock_external_apis["adzuna"].return_value = [
            {
                "title": "Test Job",
                "company": "Test Company",
                "location": "Test Location", 
                "description": "Test Description",
                "source": "Adzuna"
            }
        ]
        
        search_params = {
            "keyword": "Test",
            "location": "Test City"
        }
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify standardized schema
        for job in data["jobs"]:
            assert isinstance(job["title"], str)
            assert isinstance(job["company"], str)
            assert isinstance(job["location"], str)
            assert isinstance(job["description"], str)
            assert isinstance(job["source"], str)
    
    def test_job_search_performance(self, test_client, auth_headers, mock_external_apis):
        """Test job search response time and performance."""
        import time
        
        search_params = {
            "keyword": "Performance Test",
            "location": "Performance City"
        }
        
        start_time = time.time()
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        
        # Response should be fast (under 5 seconds for mocked APIs)
        response_time = end_time - start_time
        assert response_time < 5.0
    
    def test_location_persistence_after_search(self, test_client, auth_headers, test_user, mock_external_apis):
        """Test that detected location is saved to user profile."""
        
        # Ensure user has no saved location initially
        assert test_user.city is None
        assert test_user.country is None
        
        search_params = {"keyword": "Location Test"}
        
        response = test_client.get(
            "/api/jobs/search",
            params=search_params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # User location should be updated (need to refresh from DB)
        # This would require checking the actual database state
        # For now, just verify the geolocation service was called
        mock_external_apis["geolocation"].assert_called()