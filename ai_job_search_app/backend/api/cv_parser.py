from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from logging import getLogger
from sqlalchemy.orm import Session
from .. import schemas
from ..models.db.database import get_db
from .auth import get_current_active_user
from ..services.ml_service import ml_service
from ..utils.cv_utils import extract_text_from_pdf, extract_text_from_docx
import io
import re
import spacy
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
import numpy as np
import cv2
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Check if Tesseract is available
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    # Try to get Tesseract version to check if it's properly installed
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    # Logger not available at module level, will warn later
    pass

# Load spaCy models for different languages
try:
    nlp_en = spacy.load("en_core_web_sm")
except IOError:
    nlp_en = None

# Try to load other language models if available
nlp_models = {"en": nlp_en}
try:
    nlp_de = spacy.load("de_core_news_sm")
    nlp_models["de"] = nlp_de
except IOError:
    pass

try:
    nlp_fr = spacy.load("fr_core_news_sm")
    nlp_models["fr"] = nlp_fr
except IOError:
    pass

try:
    nlp_es = spacy.load("es_core_news_sm")
    nlp_models["es"] = nlp_es
except IOError:
    pass

logger = getLogger(__name__)
router = APIRouter()

class CVParser:
    """Advanced CV parsing service using NLP, OCR, and ML techniques with multilingual support"""
    
    def __init__(self):
        self.skill_patterns = self._load_skill_patterns()
        self.education_patterns = self._load_education_patterns()
        self.experience_patterns = self._load_experience_patterns()
        self.supported_languages = ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'sv', 'da', 'no']
        
        # Warn about Tesseract availability
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract OCR not available. Image processing will be limited.")
    
    def _load_skill_patterns(self) -> List[str]:
        """Load comprehensive skill patterns for extraction"""
        return [
            # Programming Languages
            r'\b(?:python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust|swift|kotlin|scala|r|matlab)\b',
            # Web Technologies
            r'\b(?:react|vue|angular|node\.js|express|django|flask|fastapi|spring|laravel|rails)\b',
            # Databases
            r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|oracle|sql server|sqlite|cassandra)\b',
            # Cloud & DevOps
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|terraform|ansible|chef|puppet)\b',
            # Data Science & ML
            r'\b(?:pandas|numpy|scikit-learn|tensorflow|pytorch|keras|matplotlib|seaborn|plotly)\b',
            # Tools & Frameworks
            r'\b(?:git|github|gitlab|jira|confluence|slack|figma|adobe|photoshop|illustrator)\b',
            # Methodologies
            r'\b(?:agile|scrum|kanban|devops|ci/cd|tdd|bdd|microservices|rest|graphql)\b',
            # Academic & Research
            r'\b(?:research|analysis|writing|teaching|supervision|publication|thesis|dissertation)\b'
        ]
    
    def _load_education_patterns(self) -> Dict[str, str]:
        """Load patterns for education extraction with multilingual support"""
        return {
            'degree': r'\b(?:bachelor|master|phd|doctorate|associate|diploma|certificate|b\.?[sa]\.?|m\.?[sa]\.?|ph\.?d\.?|professor|lecturer|dr\.?)\b',
            'institution': r'(?:university|college|institute|school|academy|université|universität|universidad)',
            'year': r'\b(?:19|20)\d{2}\b'
        }
    
    def _load_experience_patterns(self) -> Dict[str, str]:
        """Load patterns for experience extraction"""
        return {
            'role': r'\b(?:developer|engineer|manager|analyst|designer|consultant|architect|lead|senior|junior|professor|lecturer|assistant|researcher)\b',
            'company': r'(?:inc|ltd|llc|corp|corporation|company|technologies|systems|solutions|university|college)',
            'duration': r'\b(?:\d{1,2})\s*(?:year|month|yr|mo)s?\b'
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        try:
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = ' '.join(clean_text.split())
            
            if len(clean_text) < 50:
                return 'en'  # Default to English for short texts
            
            detected_lang = detect(clean_text)
            return detected_lang if detected_lang in self.supported_languages else 'en'
        except LangDetectException:
            return 'en'  # Default to English if detection fails
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(thresh)
        
        return processed_image
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image using OCR with multilingual support"""
        if not TESSERACT_AVAILABLE:
            raise ValueError(
                "Tesseract OCR is not installed. Please install Tesseract OCR:\n"
                "Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "Linux: sudo apt-get install tesseract-ocr\n"
                "macOS: brew install tesseract"
            )
        
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image for better OCR
            processed_image = self.preprocess_image(image)
            
            # Configure Tesseract for multiple languages
            # Start with English and common European languages
            languages = 'eng+deu+fra+spa+ita+por+nld+swe+dan+nor'
            
            # Extract text using Tesseract OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                processed_image, 
                lang=languages,
                config=custom_config
            )
            
            # Clean extracted text
            text = self._clean_ocr_text(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            raise ValueError(f"Failed to extract text from image: {str(e)}")
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')  # In names/words
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix split numbers
        
        # Remove lines with only special characters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if re.search(r'[a-zA-Z]', line):  # Keep lines with letters
                cleaned_lines.append(line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def extract_skills(self, text: str, language: str = 'en') -> List[str]:
        """Extract skills from CV text using advanced NLP and pattern matching"""
        skills = set()
        text_lower = text.lower()
        
        # Pattern-based extraction
        for pattern in self.skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.update(matches)
        
        # NLP-based extraction using appropriate language model
        nlp = nlp_models.get(language, nlp_models.get('en'))
        if nlp:
            try:
                doc = nlp(text)
                # Extract technical terms and proper nouns that might be skills
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        len(token.text) > 2 and 
                        token.text.lower() not in ['experience', 'education', 'skills', 'work', 'job']):
                        # Check if it looks like a technical skill
                        if any(keyword in token.text.lower() for keyword in 
                               ['script', 'ware', 'base', 'end', 'stack', 'tech', 'dev', 'api']):
                            skills.add(token.text.lower())
            except Exception as e:
                logger.warning(f"NLP processing failed for language {language}: {e}")
        
        # Manual pattern matching for complex skills
        complex_patterns = [
            r'\b(?:node\.?js|react\.?js|vue\.?js|angular\.?js)\b',
            r'\b(?:c\+\+|c#|\.net)\b',
            r'\b(?:ci/cd|devops|mlops)\b',
            r'\b(?:rest\s*api|graphql|soap)\b'
        ]
        
        for pattern in complex_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                skills.add(match.lower())
        
        # Clean and filter skills
        cleaned_skills = []
        for skill in skills:
            skill = skill.strip().lower()
            if len(skill) > 1 and (skill.isalpha() or '.' in skill or '+' in skill or '#' in skill):
                cleaned_skills.append(skill)
        
        return list(set(cleaned_skills))[:25]  # Limit to top 25 skills
    
    def extract_education(self, text: str, language: str = 'en') -> List[schemas.Education]:
        """Extract education information from CV text with multilingual support"""
        education_list = []
        lines = text.split('\n')
        
        current_education = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for degree patterns
            degree_match = re.search(self.education_patterns['degree'], line, re.IGNORECASE)
            if degree_match:
                if current_education:
                    education_list.append(schemas.Education(**current_education))
                current_education = {'degree': degree_match.group()}
            
            # Look for institution patterns
            institution_match = re.search(self.education_patterns['institution'], line, re.IGNORECASE)
            if institution_match and 'degree' in current_education:
                current_education['institution'] = line
            
            # Look for years
            year_matches = re.findall(self.education_patterns['year'], line)
            if year_matches and 'degree' in current_education:
                current_education['years'] = ' - '.join(year_matches)
        
        if current_education:
            education_list.append(schemas.Education(**current_education))
        
        return education_list[:5]  # Limit to 5 education entries
    
    def extract_experience(self, text: str, language: str = 'en') -> List[schemas.Experience]:
        """Extract work experience from CV text with multilingual support"""
        experience_list = []
        
        # Split text into sections
        sections = re.split(r'\n\s*\n', text)
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip short sections
                continue
            
            # Look for role indicators
            role_match = re.search(self.experience_patterns['role'], section, re.IGNORECASE)
            if role_match:
                lines = section.split('\n')
                experience = {}
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract role from first line with role pattern
                    if 'role' not in experience and re.search(self.experience_patterns['role'], line, re.IGNORECASE):
                        experience['role'] = line
                    
                    # Extract company
                    elif 'company' not in experience and re.search(self.experience_patterns['company'], line, re.IGNORECASE):
                        experience['company'] = line
                    
                    # Extract duration/period
                    elif 'period' not in experience and re.search(self.experience_patterns['duration'], line, re.IGNORECASE):
                        experience['period'] = line
                
                # Use section as description if we found role indicators
                if 'role' in experience:
                    experience['description'] = section[:500]  # Limit description length
                    experience_list.append(schemas.Experience(**experience))
        
        return experience_list[:5]  # Limit to 5 experience entries
    
    def extract_summary(self, text: str, language: str = 'en') -> str:
        """Extract or generate a professional summary with multilingual support"""
        lines = text.split('\n')
        
        # Look for summary section (multilingual keywords)
        summary_keywords = ['summary', 'profile', 'objective', 'overview', 'about', 'bio', 'biography']
        summary_section = ""
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in summary_keywords):
                # Take next few lines as summary
                summary_lines = lines[i+1:i+4]
                summary_section = ' '.join([l.strip() for l in summary_lines if l.strip()])
                break
        
        if summary_section:
            return summary_section[:300]  # Limit summary length
        
        # Generate summary from first meaningful paragraph if no explicit summary found
        first_paragraph = ""
        for line in lines[:10]:
            if line.strip() and len(line.strip()) > 20:
                first_paragraph = line.strip()
                break
        
        return first_paragraph[:300] if first_paragraph else "Professional with relevant experience and skills."
    
    def parse_cv_text(self, text: str) -> Dict[str, Any]:
        """Main method to parse CV text and extract structured data with multilingual support"""
        if not text or len(text.strip()) < 50:
            raise ValueError("CV text is too short or empty")
        
        # Detect language
        detected_language = self.detect_language(text)
        logger.info(f"Detected language: {detected_language}")
        
        # Use ML model to classify CV category
        try:
            predicted_category = ml_service.predict(text)
        except Exception as e:
            logger.warning(f"ML model prediction failed: {e}")
            predicted_category = "General"
        
        # Extract structured information
        skills = self.extract_skills(text, detected_language)
        education = self.extract_education(text, detected_language)
        experience = self.extract_experience(text, detected_language)
        summary = self.extract_summary(text, detected_language)
        
        return {
            'predicted_category': predicted_category,
            'summary': summary,
            'skills': skills,
            'educations': education,
            'experiences': experience,
            'detected_language': detected_language
        }

# Initialize CV parser
cv_parser = CVParser()

@router.post("/parse", response_model=schemas.CVParsingDetailsResult, summary="Parse a CV and save to user profile")
async def parse_cv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Uploads a CV file, parses it to extract structured data using advanced NLP techniques with OCR and multilingual support,
    and saves the result to the current user's profile.
    """
    contents = await file.read()
    content_type = file.content_type
    logger.info(f"Received file '{file.filename}' with content type '{content_type}' for user '{current_user.email}'")

    try:
        # Extract text based on file type
        if content_type.startswith("image/"):
            logger.info("Processing image file with OCR")
            cv_text = cv_parser.extract_text_from_image(contents)
        elif content_type == "application/pdf":
            cv_text = extract_text_from_pdf(contents)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            cv_text = extract_text_from_docx(io.BytesIO(contents))
        elif content_type == "text/plain":
            cv_text = contents.decode('utf-8')
        else:
            logger.warning(f"Unsupported file type uploaded: {content_type}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {content_type}. Please upload a PDF, DOCX, TXT, or image file (PNG, JPG, JPEG)."
            )

        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CV file appears to be empty or contains insufficient text for parsing."
            )

        logger.info(f"Extracted text length: {len(cv_text)} characters")

        # Parse the CV text
        parsed_data = cv_parser.parse_cv_text(cv_text)

        if not parsed_data:
            logger.error("CV parsing returned no data.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to parse CV: Parser returned no data."
            )

        # Save the parsed data to the user's profile in the database
        try:
            # Update user's parsed CV data
            from ..models.db import user as user_model
            db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
            if db_user:
                db_user.parsed_cv_data = parsed_data
                db.commit()
                db.refresh(db_user)
                logger.info(f"Successfully saved parsed CV data for user '{current_user.email}'")
            else:
                raise Exception("User not found in database")
                
        except Exception as db_error:
            logger.error(f"Database error saving parsed CV data for user '{current_user.email}': {db_error}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save parsed CV data to user profile."
            )

        # Add filename to the response
        parsed_data['filename'] = file.filename
        return schemas.CVParsingDetailsResult(**parsed_data)

    except ValueError as e:
        logger.error(f"ValueError during CV parsing: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during CV parsing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during CV parsing."
        ) 