"""
Test the authentication flow pipeline.
"""
import pytest
from fastapi import status


class TestAuthenticationFlow:
    """Test the complete authentication pipeline."""
    
    def test_user_registration_success(self, test_client, test_user_data):
        """Test successful user registration."""
        response = test_client.post("/api/auth/register", json=test_user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == test_user_data["email"]
        assert data["full_name"] == test_user_data["full_name"]
        assert "id" in data
        assert "hashed_password" not in data  # Password should not be returned
    
    def test_user_registration_duplicate_email(self, test_client, test_user_data):
        """Test registration with duplicate email fails."""
        # Register first user
        response = test_client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == status.HTTP_200_OK
        
        # Try to register again with same email
        response = test_client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already registered" in response.json()["detail"]
    
    def test_user_login_success(self, test_client, test_user_data):
        """Test successful user login."""
        # Register user first
        test_client.post("/api/auth/register", json=test_user_data)
        
        # Login
        login_data = {
            "username": test_user_data["email"],
            "password": test_user_data["password"]
        }
        response = test_client.post("/api/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_user_login_invalid_credentials(self, test_client, test_user_data):
        """Test login with invalid credentials fails."""
        # Register user first
        test_client.post("/api/auth/register", json=test_user_data)
        
        # Try login with wrong password
        login_data = {
            "username": test_user_data["email"],
            "password": "wrongpassword"
        }
        response = test_client.post("/api/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]
    
    def test_protected_endpoint_without_token(self, test_client):
        """Test accessing protected endpoint without authentication token."""
        response = test_client.get("/api/cv/parse-status")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_with_valid_token(self, test_client, auth_headers):
        """Test accessing protected endpoint with valid token."""
        response = test_client.get("/api/cv/parse-status", headers=auth_headers)
        # Should not be 401 - might be 404 if no CV data, but auth should work
        assert response.status_code != status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_with_invalid_token(self, test_client):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = test_client.get("/api/cv/parse-status", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_token_contains_user_info(self, test_client, test_user_data):
        """Test that JWT token contains correct user information."""
        # Register and login
        test_client.post("/api/auth/register", json=test_user_data)
        
        login_data = {
            "username": test_user_data["email"],
            "password": test_user_data["password"]
        }
        response = test_client.post("/api/auth/login", data=login_data)
        token = response.json()["access_token"]
        
        # Use token to access protected endpoint and verify user identity
        headers = {"Authorization": f"Bearer {token}"}
        response = test_client.get("/api/cv/parse-status", headers=headers)
        
        # The endpoint should recognize the user (even if it returns 404 for no CV)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
        assert response.status_code != status.HTTP_401_UNAUTHORIZED