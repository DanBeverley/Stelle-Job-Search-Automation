#!/usr/bin/env python3
"""
Test runner script for AI Job Search Application.
Run comprehensive tests across all components of the application pipeline.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def setup_test_environment():
    """Setup test environment and dependencies."""
    print("🔧 Setting up test environment...")
    
    # Ensure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Install test requirements (optional - user should do this manually)
    # run_command("pip install -r test_requirements.txt", "Installing test dependencies")
    
    # Set environment variables for testing
    os.environ.update({
        "TESTING": "true",
        "SECRET_KEY": "test-secret-key-for-jwt-tokens-12345678901234567890123456789012",
        "ENCRYPTION_KEY": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "sqlite:///./test.db"
    })


def run_tests(test_suite=None, verbose=False, coverage=True, parallel=False):
    """Run the test suite."""
    
    setup_test_environment()
    
    # Base pytest command
    cmd_parts = ["python", "-m", "pytest", "ai_job_search_app/tests/"]
    
    # Add specific test suite if specified
    if test_suite:
        if test_suite == "auth":
            cmd_parts.append("ai_job_search_app/tests/test_auth_flow.py")
        elif test_suite == "cv":
            cmd_parts.append("ai_job_search_app/tests/test_cv_processing_pipeline.py")
        elif test_suite == "job_search":
            cmd_parts.append("ai_job_search_app/tests/test_job_search_pipeline.py")
        elif test_suite == "ai_features":
            cmd_parts.append("ai_job_search_app/tests/test_ai_features_pipeline.py")
        elif test_suite == "application_tracking":
            cmd_parts.append("ai_job_search_app/tests/test_application_tracking_pipeline.py")
        elif test_suite == "integration":
            cmd_parts.append("ai_job_search_app/tests/test_integration_pipeline.py")
    
    # Add options
    if verbose:
        cmd_parts.extend(["-v", "-s"])
    
    if coverage:
        cmd_parts.extend([
            "--cov=ai_job_search_app/backend",
            "--cov-report=html:test_coverage",
            "--cov-report=term-missing",
            "--cov-fail-under=70"
        ])
    
    if parallel:
        cmd_parts.extend(["-n", "auto"])  # Requires pytest-xdist
    
    # Add output options
    cmd_parts.extend([
        "--html=test_results.html",
        "--self-contained-html",
        "--tb=short"
    ])
    
    command = " ".join(cmd_parts)
    return run_command(command, f"Running {test_suite or 'all'} tests")


def run_linting():
    """Run code linting and style checks."""
    linting_commands = [
        ("flake8 ai_job_search_app/backend/ --max-line-length=120 --ignore=E501,W503", "Code style check with flake8"),
        ("black --check ai_job_search_app/backend/", "Code formatting check with black"),
        ("isort --check-only ai_job_search_app/backend/", "Import sorting check with isort"),
    ]
    
    all_passed = True
    for command, description in linting_commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def run_security_checks():
    """Run security vulnerability checks."""
    security_commands = [
        ("bandit -r ai_job_search_app/backend/ -f json -o security_report.json", "Security vulnerability scan with bandit"),
        ("safety check --json --output safety_report.json", "Dependency vulnerability check with safety"),
    ]
    
    all_passed = True
    for command, description in security_commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="AI Job Search App Test Runner")
    parser.add_argument("--suite", choices=["auth", "cv", "job_search", "ai_features", "application_tracking", "integration"], 
                       help="Run specific test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all checks (tests, linting, security)")
    
    args = parser.parse_args()
    
    print("🚀 AI Job Search Application Test Runner")
    print("=" * 60)
    
    success = True
    
    # Run tests
    if not args.lint and not args.security:
        success &= run_tests(
            test_suite=args.suite,
            verbose=args.verbose,
            coverage=not args.no_coverage,
            parallel=args.parallel
        )
    
    # Run linting if requested
    if args.lint or args.all:
        success &= run_linting()
    
    # Run security checks if requested
    if args.security or args.all:
        success &= run_security_checks()
    
    # Final results
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("📊 Test report: test_results.html")
        if not args.no_coverage:
            print("📈 Coverage report: test_coverage/index.html")
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main()