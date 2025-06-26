# AI Job Search Automation

Production-ready FastAPI application that automates job searching with AI-powered features including CV parsing, job matching, salary prediction, and automated cover letter generation.

## 🚀 Features

### Core Functionality
- **CV Processing**: Multi-format parsing (PDF/DOCX/Images) with OCR and NLP extraction
- **Job Search**: Multi-provider aggregation (Adzuna, TheirStack) with location detection
- **User Authentication**: JWT-based secure authentication with encrypted data storage
- **Application Tracking**: Complete CRUD system for job application management

### AI-Powered Features
- **Salary Prediction**: XGBoost model trained on 30K+ records (RMSE: $13,782)
- **Skill Analysis**: Gap analysis with automated course recommendations
- **Cover Letter Generation**: Fine-tuned Qwen2.5-3B model with personalization
- **Interview Preparation**: DialoGPT model for question generation and STAR analysis

### System Features
- **Centralized Configuration**: Environment-based settings with validation
- **Comprehensive Testing**: 59+ tests covering complete pipeline
- **Health Monitoring**: Real-time system status and configuration validation
- **Production Ready**: Logging, error handling, and security best practices

## 🛠️ Quick Start

### 1. Installation
```bash
git clone <repository>
cd Stelle-Job-Search-Automation
pip install -r requirements.txt
```

### 2. Configuration
Set up your `.env` file with required API keys:
```bash
SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-aes-encryption-key
ADZUNA_APP_ID=your-adzuna-id
ADZUNA_APP_KEY=your-adzuna-key
THEIRSTACK_API_KEY=your-theirstack-key
```

### 3. Testing
```bash
# Run comprehensive tests
python run_tests.py --all

# Run specific test suites
python run_tests.py --suite integration
```

### 4. Start Application
```bash
uvicorn ai_job_search_app.backend.main:app --reload
```

## 📊 API Endpoints

- **Authentication**: `/api/auth/` - User registration/login
- **CV Processing**: `/api/cv/` - Upload and parse CVs
- **Job Search**: `/api/jobs/` - Multi-provider job search
- **AI Features**: `/api/salary/`, `/api/skills/`, `/api/cover-letter/`, `/api/interview/`
- **Applications**: `/api/applications/` - Track job applications
- **Health Check**: `/api/health/` - System status

## 🧪 Testing

Comprehensive test suite with 59+ tests covering:
- Authentication flow and security
- CV processing pipeline (PDF/DOCX/OCR)
- Job search and provider integration
- AI features (salary, skills, cover letters, interviews)
- Application tracking and data persistence
- End-to-end user journey validation

## 📁 Architecture

```
ai_job_search_app/
├── backend/
│   ├── config/settings.py      # Centralized configuration
│   ├── api/                    # REST API endpoints
│   ├── services/               # Business logic & ML models
│   ├── models/                 # Database models
│   └── utils/                  # Common utilities
├── tests/                      # Comprehensive test suite
├── .env                        # Environment configuration
└── run_tests.py               # Test runner
```

## 🔧 ML Models

- **Salary Prediction**: XGBoost regression on LinkedIn/JobPosting datasets
- **CV Classification**: BERT model for resume categorization
- **Cover Letter**: Fine-tuned Qwen2.5-3B with QLoRA optimization
- **Interview Prep**: DialoGPT-medium with LoRA for question generation

All models support both CPU and GPU execution with automatic device detection.