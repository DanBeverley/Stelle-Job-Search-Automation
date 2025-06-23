# AI Job Search Automation

FastAPI-based application that automates job searching with AI-powered features including CV parsing, job matching, salary prediction, and automated cover letter generation.

## Features

- **Job Search API**: Aggregates listings from multiple sources (Adzuna, TheirStack)
- **User Authentication**: JWT-based secure authentication system
- **Salary Prediction**: ML model trained on 30K+ job records (RMSE: $13,782)
- **Cover Letter Generation**: Fine-tuned Llama-2-7B model
- **Interview Prep**: Fine-tuned Mistral-7B model for question generation
- **Geolocation**: Automatic location detection for job searches

## Training Pipeline

The application includes ML models trained using state-of-the-art techniques:

- **Salary Model**: XGBoost trained on LinkedIn, JobPosting, and Azrai datasets
- **Cover Letter Model**: Llama-2-7B-Chat with QLoRA optimization
- **Interview Model**: Mistral-7B-Instruct with QLoRA optimization

Training scripts support Kaggle environments with automatic dataset loading and model optimization.