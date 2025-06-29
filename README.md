# AI Job Search Automation

Full-stack AI-powered job search application with automated CV analysis, intelligent job matching, salary prediction, and personalized application assistance.

## 🚀 Current Status

### ✅ Completed Features
- **Full-Stack Architecture**: FastAPI backend (port 8000) + React frontend (port 3000)
- **CV Processing**: Multi-format parsing (PDF/DOCX/Images) with OCR and NLP extraction
- **Job Search Integration**: Adzuna and TheirStack API providers with location detection
- **User Authentication**: JWT-based secure authentication with encrypted data storage
- **Application Tracking**: Complete CRUD system for job application management
- **AI Models**: Working salary prediction, cover letter generation, and interview preparation
- **Database**: SQLite integration with user data persistence
- **Security**: AES-256 encryption for sensitive data, proper authentication flows
- **Testing**: Comprehensive test suite with 100% ML pipeline success rate

### 🔧 System Capabilities
- **Salary Prediction**: Scikit-learn model with realistic predictions ($30K-$500K range)
- **Cover Letter Generation**: Fine-tuned language model with company-specific personalization
- **Interview Preparation**: Automated response generation with length validation
- **Skill Analysis**: NLP-based skill extraction and gap analysis
- **Multi-Provider Job Search**: Aggregated results from multiple job search APIs
- **Real-time Health Monitoring**: API status and configuration validation

## 🛠️ Quick Start

### 1. Prerequisites
```bash
git clone https://github.com/your-repo/Stelle-Job-Search-Automation.git
cd Stelle-Job-Search-Automation
pip install -r requirements.txt
```

### 2. Configuration (Optional)
For full functionality, configure API keys in backend directory:
```bash
# Optional: Set up external API keys for enhanced features
SECRET_KEY=your-jwt-secret-key
ADZUNA_APP_ID=your-adzuna-id  
ADZUNA_APP_KEY=your-adzuna-key
THEIRSTACK_API_KEY=your-theirstack-key
```

### 3. Start Full-Stack Application
```bash
# One-command startup (recommended)
python3 start_fullstack.py

# Manual startup
# Backend: python3 -m uvicorn ai_job_search_app.backend.main:app --port 8000
# Frontend: cd ai_job_search_app/frontend && npm start
```

### 4. Access Application
- **Frontend**: http://localhost:3000 (React application)
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Backend API**: http://localhost:8000 (REST endpoints)

### 5. Testing
```bash
# Run comprehensive test suite
python run_tests.py

# Test specific components  
python test_pipeline.py  # ML models
python test_api_endpoints.py  # API functionality
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

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Core** | `/api/health/` | System health and status |
| **Auth** | `/api/auth/register`, `/api/auth/login` | User authentication |
| **CV** | `/api/cv/parse` | Upload and parse CVs/resumes |
| **Jobs** | `/api/jobs/search` | Multi-provider job search |
| **AI** | `/api/salary/predict` | Salary prediction model |
| **AI** | `/api/cover-letter/generate` | AI cover letter generation |
| **AI** | `/api/interview/generate-response` | Interview preparation |
| **AI** | `/api/skills/analyze` | Skill gap analysis |
| **Apps** | `/api/applications/` | Job application tracking |

## 🧪 Testing & Quality

- **Test Coverage**: 100% ML pipeline success rate  
- **Test Suites**: Authentication, CV processing, job search, AI features, application tracking
- **Performance**: All AI models respond within acceptable time limits
- **Security**: JWT authentication, AES-256 encryption, input validation
- **Error Handling**: Comprehensive logging and graceful error recovery

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