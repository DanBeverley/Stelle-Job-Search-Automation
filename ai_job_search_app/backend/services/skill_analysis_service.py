import requests
from typing import List, Dict, Any, Set
import os
import re
from cachetools import cached, TTLCache
from difflib import SequenceMatcher
import spacy
from collections import Counter

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    nlp = None

# --- edX API Configuration ---
EDX_API_CLIENT_ID = os.environ.get("EDX_API_CLIENT_ID")
EDX_API_CLIENT_SECRET = os.environ.get("EDX_API_CLIENT_SECRET")
EDX_TOKEN_URL = "https://api.edx.org/oauth2/v1/access_token"
EDX_COURSE_API_URL = "https://api.edx.org/catalog/v1/courses"

# Comprehensive skill taxonomy with categories
SKILL_TAXONOMY = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 
        'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell', 'powershell', 'vba'
    ],
    'web_technologies': [
        'html', 'css', 'react', 'vue', 'angular', 'node.js', 'express', 'django', 'flask', 
        'fastapi', 'spring', 'laravel', 'rails', 'jquery', 'bootstrap', 'sass', 'less'
    ],
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sql server', 
        'sqlite', 'cassandra', 'dynamodb', 'neo4j', 'influxdb', 'mariadb'
    ],
    'cloud_devops': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 
        'chef', 'puppet', 'gitlab ci', 'github actions', 'circleci', 'travis ci'
    ],
    'data_science_ml': [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib', 
        'seaborn', 'plotly', 'jupyter', 'spark', 'hadoop', 'tableau', 'power bi'
    ],
    'tools_frameworks': [
        'git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'figma', 'adobe', 
        'photoshop', 'illustrator', 'sketch', 'invision', 'postman', 'swagger'
    ],
    'methodologies': [
        'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd', 'microservices', 
        'rest', 'graphql', 'soap', 'oauth', 'jwt', 'solid', 'design patterns'
    ],
    'soft_skills': [
        'leadership', 'communication', 'teamwork', 'problem solving', 'project management',
        'time management', 'critical thinking', 'creativity', 'adaptability', 'mentoring'
    ]
}

# Flatten all skills for easy access
ALL_SKILLS = []
for category, skills in SKILL_TAXONOMY.items():
    ALL_SKILLS.extend(skills)

# Cache the access token for 1 hour (3600 seconds)
token_cache = TTLCache(maxsize=1, ttl=3600)

class SkillAnalyzer:
    """Advanced skill analysis service with semantic matching and intelligent recommendations"""
    
    def __init__(self):
        self.skill_synonyms = self._load_skill_synonyms()
        self.skill_patterns = self._compile_skill_patterns()
    
    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms for better matching"""
        return {
            'javascript': ['js', 'ecmascript', 'node'],
            'typescript': ['ts'],
            'python': ['py'],
            'postgresql': ['postgres', 'psql'],
            'mongodb': ['mongo'],
            'kubernetes': ['k8s'],
            'docker': ['containerization'],
            'aws': ['amazon web services'],
            'azure': ['microsoft azure'],
            'gcp': ['google cloud platform', 'google cloud'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'artificial intelligence': ['ai', 'machine learning', 'ml'],
            'react': ['reactjs', 'react.js'],
            'vue': ['vuejs', 'vue.js'],
            'angular': ['angularjs'],
            'node.js': ['nodejs', 'node'],
            'c++': ['cpp', 'c plus plus'],
            'c#': ['csharp', 'c sharp'],
            'sql': ['structured query language'],
            'nosql': ['no sql', 'non-relational'],
            'rest': ['restful', 'rest api'],
            'graphql': ['graph ql'],
            'ci/cd': ['continuous integration', 'continuous deployment', 'continuous delivery']
        }
    
    def _compile_skill_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for skill extraction"""
        patterns = []
        for skill in ALL_SKILLS:
            # Create pattern that matches skill with word boundaries
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill name using synonyms and common variations"""
        skill_lower = skill.lower().strip()
        
        # Check for exact matches in synonyms
        for main_skill, synonyms in self.skill_synonyms.items():
            if skill_lower == main_skill or skill_lower in synonyms:
                return main_skill
        
        # Check for partial matches with high similarity
        for main_skill in ALL_SKILLS:
            if self.similarity(skill_lower, main_skill) > 0.85:
                return main_skill
        
        return skill_lower
    
    def extract_skills_from_text(self, text: str) -> Set[str]:
        """Extract skills from text using multiple techniques"""
        skills = set()
        text_lower = text.lower()
        
        # Pattern-based extraction
        for pattern in self.skill_patterns:
            matches = pattern.findall(text_lower)
            skills.update(matches)
        
        # NLP-based extraction using spaCy
        if nlp:
            doc = nlp(text)
            
            # Extract entities that might be technologies
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
                    normalized = self.normalize_skill(ent.text)
                    if normalized in ALL_SKILLS:
                        skills.add(normalized)
            
            # Extract noun phrases that might be skills
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to short phrases
                    normalized = self.normalize_skill(chunk.text)
                    if normalized in ALL_SKILLS:
                        skills.add(normalized)
        
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
                normalized = self.normalize_skill(match)
                skills.add(normalized)
        
        return skills
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills by type"""
        categorized = {category: [] for category in SKILL_TAXONOMY.keys()}
        
        for skill in skills:
            skill_lower = skill.lower()
            for category, category_skills in SKILL_TAXONOMY.items():
                if skill_lower in category_skills:
                    categorized[category].append(skill)
                    break
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def calculate_skill_match_score(self, user_skills: List[str], required_skills: List[str]) -> float:
        """Calculate a match score between user skills and required skills"""
        if not required_skills:
            return 1.0
        
        user_skills_set = set(skill.lower() for skill in user_skills)
        required_skills_set = set(skill.lower() for skill in required_skills)
        
        # Direct matches
        direct_matches = len(user_skills_set.intersection(required_skills_set))
        
        # Semantic matches (using synonyms and similarity)
        semantic_matches = 0
        for req_skill in required_skills_set:
            if req_skill not in user_skills_set:
                for user_skill in user_skills_set:
                    if self.similarity(req_skill, user_skill) > 0.8:
                        semantic_matches += 0.5  # Partial credit for semantic matches
                        break
        
        total_matches = direct_matches + semantic_matches
        return min(total_matches / len(required_skills_set), 1.0)
    
    def prioritize_missing_skills(self, missing_skills: List[str], job_description: str) -> List[str]:
        """Prioritize missing skills based on frequency and importance in job description"""
        skill_importance = Counter()
        
        # Count frequency of each skill in job description
        job_desc_lower = job_description.lower()
        for skill in missing_skills:
            count = len(re.findall(r'\b' + re.escape(skill.lower()) + r'\b', job_desc_lower))
            skill_importance[skill] = count
        
        # Sort by importance (frequency) and return
        return [skill for skill, _ in skill_importance.most_common()]

# Initialize skill analyzer
skill_analyzer = SkillAnalyzer()

@cached(token_cache)
def get_edx_access_token() -> str:
    """
    Retrieves an access token from the edX API using client credentials.
    The token is cached to avoid repeated requests.
    """
    if not EDX_API_CLIENT_ID or not EDX_API_CLIENT_SECRET:
        raise ValueError("edX API credentials not configured. Please set EDX_API_CLIENT_ID and EDX_API_CLIENT_SECRET environment variables.")
    
    payload = {
        'grant_type': 'client_credentials',
        'client_id': EDX_API_CLIENT_ID,
        'client_secret': EDX_API_CLIENT_SECRET,
        'token_type': 'jwt'
    }
    response = requests.post(EDX_TOKEN_URL, data=payload, timeout=10)
    response.raise_for_status()
    return response.json().get('access_token')

def get_course_recommendations(skill: str, max_courses: int = 3) -> List[Dict[str, Any]]:
    """
    Fetches course recommendations for a specific skill from the edX API.
    """
    try:
        access_token = get_edx_access_token()
        headers = {'Authorization': f'JWT {access_token}'}
        
        # Try multiple search queries for better results
        search_queries = [skill, f"{skill} course", f"{skill} tutorial", f"learn {skill}"]
        all_courses = []
        
        for query in search_queries:
            params = {'search_query': query, 'size': max_courses}
            response = requests.get(EDX_COURSE_API_URL, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            for course in response.json().get('results', []):
                course_data = {
                    "title": course.get('title', 'N/A'),
                    "url": course.get('marketing_url', ''),
                    "organization": course.get('owners', [{}])[0].get('name', 'N/A'),
                    "short_description": course.get('short_description', '')[:200],
                    "level": course.get('level_type', 'N/A'),
                    "skill": skill
                }
                
                # Avoid duplicates
                if not any(existing['title'] == course_data['title'] for existing in all_courses):
                    all_courses.append(course_data)
                
                if len(all_courses) >= max_courses:
                    break
            
            if len(all_courses) >= max_courses:
                break
        
        return all_courses[:max_courses]
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching courses for '{skill}': {e}")
        # Return fallback recommendations
        return [{
            "title": f"Learn {skill.title()}",
            "url": f"https://www.edx.org/search?q={skill}",
            "organization": "edX",
            "short_description": f"Find courses related to {skill}",
            "level": "All Levels",
            "skill": skill
        }]
    except Exception as e:
        print(f"Unexpected error fetching courses for '{skill}': {e}")
        return []

def analyze_skill_gap(job_description: str, user_skills: List[str]) -> Dict[str, Any]:
    """
    Analyzes the gap between user skills and job requirements with advanced matching and recommendations.
    """
    # Extract skills from job description
    required_skills = list(skill_analyzer.extract_skills_from_text(job_description))
    
    # Normalize user skills
    normalized_user_skills = [skill_analyzer.normalize_skill(skill) for skill in user_skills]
    normalized_user_skills = list(set(normalized_user_skills))  # Remove duplicates
    
    # Find matches and gaps
    user_skills_set = set(skill.lower() for skill in normalized_user_skills)
    required_skills_set = set(skill.lower() for skill in required_skills)
    
    # Direct matches
    matched_skills = [skill for skill in required_skills if skill.lower() in user_skills_set]
    
    # Missing skills
    missing_skills = [skill for skill in required_skills if skill.lower() not in user_skills_set]
    
    # Calculate match score
    match_score = skill_analyzer.calculate_skill_match_score(normalized_user_skills, required_skills)
    
    # Prioritize missing skills
    prioritized_missing = skill_analyzer.prioritize_missing_skills(missing_skills, job_description)
    
    # Categorize skills
    user_skill_categories = skill_analyzer.categorize_skills(normalized_user_skills)
    required_skill_categories = skill_analyzer.categorize_skills(required_skills)
    missing_skill_categories = skill_analyzer.categorize_skills(prioritized_missing)
    
    # Get course recommendations for top missing skills
    recommended_courses = []
    for skill in prioritized_missing[:5]:  # Limit to top 5 missing skills
        courses = get_course_recommendations(skill, max_courses=2)
        recommended_courses.extend(courses)
    
    return {
        "matched_skills": matched_skills,
        "missing_skills": prioritized_missing,
        "match_score": round(match_score * 100, 1),  # Convert to percentage
        "user_skill_categories": user_skill_categories,
        "required_skill_categories": required_skill_categories,
        "missing_skill_categories": missing_skill_categories,
        "recommended_courses": recommended_courses,
        "analysis_summary": {
            "total_required_skills": len(required_skills),
            "total_matched_skills": len(matched_skills),
            "total_missing_skills": len(prioritized_missing),
            "skill_coverage": f"{len(matched_skills)}/{len(required_skills)} skills matched"
        }
    } 