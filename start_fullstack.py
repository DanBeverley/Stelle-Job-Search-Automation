#!/usr/bin/env python3
"""
Start both backend and frontend for full-stack testing
"""
import subprocess
import sys
import os
import time
import logging
import signal
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullStackManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent
        self.backend_path = self.project_root / "ai_job_search_app" / "backend"
        self.frontend_path = self.project_root / "ai_job_search_app" / "frontend"
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")
        
        # Check if backend files exist
        if not (self.backend_path / "main.py").exists():
            logger.error("Backend main.py not found")
            return False
            
        # Check if frontend files exist
        if not (self.frontend_path / "package.json").exists():
            logger.error("Frontend package.json not found")
            return False
            
        # Check if node_modules exist
        if not (self.frontend_path / "node_modules").exists():
            logger.warning("Node modules not found, will need to install")
            
        logger.info("✅ Prerequisites check passed")
        return True
    
    def setup_environment(self):
        """Set up environment variables"""
        logger.info("Setting up environment...")
        
        # Set Python path
        env = os.environ.copy()
        backend_parent = str(self.backend_path.parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{backend_parent}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = backend_parent
            
        # Basic environment variables for FastAPI
        env.update({
            'ENVIRONMENT': 'development',
            'SECRET_KEY': 'dev-secret-key-for-testing-only',
            'DATABASE_URL': 'sqlite:///./test.db',
            'CORS_ENABLED': 'true',
            'API_DOCS_ENABLED': 'true',
            'LOG_LEVEL': 'INFO'
        })
        
        return env
    
    def install_frontend_deps(self):
        """Install frontend dependencies if needed"""
        if not (self.frontend_path / "node_modules").exists():
            logger.info("Installing frontend dependencies...")
            try:
                subprocess.run(
                    ["npm", "install"],
                    cwd=self.frontend_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("✅ Frontend dependencies installed")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install frontend dependencies: {e}")
                return False
        else:
            logger.info("✅ Frontend dependencies already installed")
            return True
    
    def start_backend(self, env):
        """Start the FastAPI backend"""
        logger.info("Starting backend server...")
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
                cwd=self.backend_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start a thread to read backend output
            def read_backend_output():
                for line in iter(self.backend_process.stdout.readline, ''):
                    print(f"[BACKEND] {line.strip()}")
                    
            backend_thread = threading.Thread(target=read_backend_output, daemon=True)
            backend_thread.start()
            
            logger.info("✅ Backend server starting on http://localhost:8000")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend"""
        logger.info("Starting frontend server...")
        
        try:
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.frontend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=dict(os.environ, BROWSER="none")  # Don't auto-open browser
            )
            
            # Start a thread to read frontend output
            def read_frontend_output():
                for line in iter(self.frontend_process.stdout.readline, ''):
                    print(f"[FRONTEND] {line.strip()}")
                    
            frontend_thread = threading.Thread(target=read_frontend_output, daemon=True)
            frontend_thread.start()
            
            logger.info("✅ Frontend server starting on http://localhost:3000")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start frontend: {e}")
            return False
    
    def wait_for_services(self):
        """Wait for services to be ready"""
        logger.info("Waiting for services to start...")
        
        # Wait a bit for services to initialize
        time.sleep(5)
        
        # Check if backend is responding
        try:
            import requests
            response = requests.get("http://localhost:8000/", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Backend is responding")
            else:
                logger.warning(f"Backend responded with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not verify backend status: {e}")
        
        logger.info("🚀 Full-stack application is ready!")
        logger.info("📖 Backend API docs: http://localhost:8000/docs")
        logger.info("🌐 Frontend application: http://localhost:3000")
        
    def cleanup(self):
        """Clean up processes"""
        logger.info("Shutting down services...")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                
        logger.info("✅ Services shut down")
    
    def run(self):
        """Run the full-stack application"""
        try:
            if not self.check_prerequisites():
                return False
                
            if not self.install_frontend_deps():
                return False
                
            env = self.setup_environment()
            
            if not self.start_backend(env):
                return False
                
            # Give backend time to start
            time.sleep(3)
            
            if not self.start_frontend():
                return False
                
            self.wait_for_services()
            
            # Keep running
            logger.info("Press Ctrl+C to stop the servers")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                
        except Exception as e:
            logger.error(f"Error running full-stack app: {e}")
            return False
        finally:
            self.cleanup()
            
        return True

def main():
    print("🚀 Starting AI Job Search Full-Stack Application...")
    
    manager = FullStackManager()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        manager.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    success = manager.run()
    if not success:
        print("❌ Failed to start full-stack application")
        sys.exit(1)

if __name__ == "__main__":
    main()