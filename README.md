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
Set up your `.env` file in the backend directory with required API keys:
```bash
cd ai_job_search_app/backend
cp .env.example .env  # Edit with your values

# Required configuration
SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-aes-encryption-key
ADZUNA_APP_ID=your-adzuna-id
ADZUNA_APP_KEY=your-adzuna-key
THEIRSTACK_API_KEY=your-theirstack-key
```

### 3. Start Backend Server
```bash
cd ai_job_search_app/backend
python3 -m uvicorn main:app --reload --port 8000
```

### 4. Start Frontend (Optional)
```bash
cd ai_job_search_app/frontend
npm install
npm start
```

### 5. Testing
```bash
# Run comprehensive tests
python run_tests.py --all

# Run specific test suites
python run_tests.py --suite integration
```

## 🧠 Model Training & Fine-tuning

### Salary Prediction Model
```bash
cd ai_job_search_app/scripts
python train_salary_model.py
```

### NLP Models (BERT, CV Classification)
```bash
cd ai_job_search_app/scripts
python train_nlp.py
```

### Fine-tune Cover Letter & Interview Models
```bash
cd ai_job_search_app/scripts
python run_finetuning.py --model_type cover_letter
python run_finetuning.py --model_type interview
```

### Data Preprocessing
```bash
cd ai_job_search_app/scripts
python preprocess_data.py
python quantize_model.py  # Optimize models for production
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
│   ├── utils/                  # Common utilities
│   └── main.py                 # FastAPI application entry
├── frontend/                   # React frontend (optional)
├── scripts/                    # Training & preprocessing scripts
├── data/models/               # Pre-trained models
├── final_model/               # Fine-tuned model artifacts
├── tests/                     # Comprehensive test suite
└── requirements.txt           # Python dependencies
```

## 🔧 ML Models

- **Salary Prediction**: XGBoost regression on LinkedIn/JobPosting datasets
- **CV Classification**: BERT model for resume categorization
- **Cover Letter**: Fine-tuned Qwen2.5-3B with QLoRA optimization
- **Interview Prep**: DialoGPT-medium with LoRA for question generation

All models support both CPU and GPU execution with automatic device detection.

## 🐛 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Model Loading**: Pre-trained models are loaded from `data/models/` and `final_model/`
- **API Keys**: Check `.env` file configuration in backend directory
- **Port Conflicts**: Default backend runs on port 8000, frontend on 3000

### Development
```bash
# Backend hot-reload
cd ai_job_search_app/backend
python3 -m uvicorn main:app --reload

# Frontend development
cd ai_job_search_app/frontend
npm run dev
```