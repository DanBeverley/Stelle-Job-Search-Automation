"""
Test the CV processing pipeline end-to-end.
"""
import pytest
import json
from unittest.mock import patch, Mock
from fastapi import status
from io import BytesIO


class TestCVProcessingPipeline:
    """Test the complete CV processing pipeline."""
    
    def test_cv_upload_and_parse_pdf_success(self, test_client, auth_headers, mock_ml_models):
        """Test successful PDF CV upload and parsing."""
        
        # Mock file processing
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_extract_skills, \
             patch('backend.services.cv_utils.extract_experience_with_nlp') as mock_extract_exp:
            
            mock_extract_pdf.return_value = "John Doe Software Engineer Python Machine Learning"
            mock_extract_skills.return_value = ["Python", "Machine Learning", "FastAPI"]
            mock_extract_exp.return_value = [
                {
                    "company": "Tech Corp",
                    "position": "Software Engineer", 
                    "duration": "2020-2023"
                }
            ]
            
            # Create test PDF file
            pdf_content = b"Sample PDF content"
            files = {"file": ("resume.pdf", BytesIO(pdf_content), "application/pdf")}
            
            response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "message" in data
            assert "parsing successful" in data["message"].lower()
    
    def test_cv_upload_unsupported_format(self, test_client, auth_headers):
        """Test CV upload with unsupported file format."""
        
        # Create test file with unsupported format
        unsupported_content = b"Unsupported file content"
        files = {"file": ("resume.xyz", BytesIO(unsupported_content), "application/xyz")}
        
        response = test_client.post(
            "/api/cv/upload-and-parse",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "unsupported" in response.json()["detail"].lower()
    
    def test_cv_parse_status_no_cv(self, test_client, auth_headers):
        """Test CV parse status when no CV has been uploaded."""
        response = test_client.get("/api/cv/parse-status", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    def test_cv_parse_status_with_cv(self, test_client, auth_headers, test_user_with_cv):
        """Test CV parse status when CV data exists."""
        response = test_client.get("/api/cv/parse-status", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "personal_info" in data
        assert "skills" in data
        assert "experience" in data
    
    def test_cv_image_ocr_processing(self, test_client, auth_headers, mock_ml_models):
        """Test CV processing with image OCR."""
        
        with patch('backend.services.cv_utils.extract_text_from_image_ocr') as mock_ocr, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_extract_skills:
            
            mock_ocr.return_value = "JOHN DOE\nSOFTWARE ENGINEER\nPython, Machine Learning"
            mock_extract_skills.return_value = ["Python", "Machine Learning"]
            
            # Create test image file
            image_content = b"Fake image content"
            files = {"file": ("resume.png", BytesIO(image_content), "image/png")}
            
            response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            mock_ocr.assert_called_once()
    
    def test_cv_multilingual_processing(self, test_client, auth_headers, mock_ml_models):
        """Test CV processing with non-English content."""
        
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf, \
             patch('langdetect.detect') as mock_lang_detect, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_extract_skills:
            
            mock_extract_pdf.return_value = "Jean Dupont Ingénieur Logiciel Python"
            mock_lang_detect.return_value = "fr"  # French
            mock_extract_skills.return_value = ["Python", "Ingénierie Logicielle"]
            
            pdf_content = b"French CV content"
            files = {"file": ("cv_french.pdf", BytesIO(pdf_content), "application/pdf")}
            
            response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
    
    def test_cv_skills_extraction_accuracy(self, test_client, auth_headers, mock_ml_models):
        """Test accuracy of skills extraction from CV."""
        
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_extract_skills:
            
            # Mock CV text with various skill formats
            cv_text = """
            John Doe
            Skills: Python, JavaScript, React.js, Machine Learning
            Technologies: Docker, Kubernetes, AWS
            Programming Languages: Java, C++, SQL
            """
            mock_extract_pdf.return_value = cv_text
            
            # Expected skills extraction
            expected_skills = [
                "Python", "JavaScript", "React.js", "Machine Learning",
                "Docker", "Kubernetes", "AWS", "Java", "C++", "SQL"
            ]
            mock_extract_skills.return_value = expected_skills
            
            pdf_content = b"CV with skills"
            files = {"file": ("skilled_resume.pdf", BytesIO(pdf_content), "application/pdf")}
            
            response = test_client.post(
                "/api/cv/upload-and-parse", 
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify skills were extracted
            status_response = test_client.get("/api/cv/parse-status", headers=auth_headers)
            cv_data = status_response.json()
            extracted_skills = cv_data.get("skills", [])
            
            # Should contain at least some of the expected skills
            assert len(extracted_skills) > 0
    
    def test_cv_ml_classification(self, test_client, auth_headers, mock_ml_models):
        """Test ML model classification of CV category."""
        
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf:
            
            mock_extract_pdf.return_value = "Software Engineer with Python and ML experience"
            
            pdf_content = b"Software engineer CV"
            files = {"file": ("engineer_resume.pdf", BytesIO(pdf_content), "application/pdf")}
            
            response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files, 
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify ML model was called for classification
            mock_ml_models["cv_classifier"].classify_cv_category.assert_called()
    
    def test_cv_parsing_error_handling(self, test_client, auth_headers):
        """Test CV parsing error handling for corrupted files."""
        
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf:
            
            # Simulate file processing error
            mock_extract_pdf.side_effect = Exception("File corrupted")
            
            pdf_content = b"Corrupted PDF content"
            files = {"file": ("corrupted.pdf", BytesIO(pdf_content), "application/pdf")}
            
            response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=auth_headers
            )
            
            # Should handle error gracefully
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    def test_cv_data_persistence(self, test_client, auth_headers, mock_ml_models):
        """Test that parsed CV data is properly persisted."""
        
        with patch('backend.services.cv_utils.extract_text_from_pdf') as mock_extract_pdf, \
             patch('backend.services.cv_utils.extract_skills_with_nlp') as mock_extract_skills:
            
            mock_extract_pdf.return_value = "Test CV content"
            mock_extract_skills.return_value = ["Test Skill"]
            
            # Upload CV
            pdf_content = b"Test CV"
            files = {"file": ("test_cv.pdf", BytesIO(pdf_content), "application/pdf")}
            
            upload_response = test_client.post(
                "/api/cv/upload-and-parse",
                files=files,
                headers=auth_headers
            )
            assert upload_response.status_code == status.HTTP_200_OK
            
            # Verify data persistence by checking status
            status_response = test_client.get("/api/cv/parse-status", headers=auth_headers)
            assert status_response.status_code == status.HTTP_200_OK
            
            cv_data = status_response.json()
            assert "skills" in cv_data
            assert len(cv_data["skills"]) > 0