#!/usr/bin/env python3
"""
Test runner script for LLM Text Summarization Tool

This script provides a convenient way to run tests with different configurations
and generate comprehensive test reports.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Please ensure pytest is installed: pip install pytest")
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=utils", "--cov-report=html", "--cov-report=term-missing"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Integration Tests")


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=utils", "--cov-report=html", "--cov-report=term-missing"])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Specific Test: {test_path}")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "performance"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Performance Tests")


def run_chinese_tests(verbose=False):
    """Run Chinese language specific tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "chinese"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Chinese Language Tests")


def run_english_tests(verbose=False):
    """Run English language specific tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "english"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "English Language Tests")


def run_visualization_tests(verbose=False):
    """Run visualization tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "visualization"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Visualization Tests")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "transformers",
        "torch",
        "scikit-learn",
        "matplotlib",
        "spacy",
        "jieba"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test runner for LLM Text Summarization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests only
  python run_tests.py --integration            # Run integration tests only
  python run_tests.py --all                    # Run all tests
  python run_tests.py --coverage               # Run with coverage report
  python run_tests.py --verbose                # Run with verbose output
  python run_tests.py --chinese                # Run Chinese language tests
  python run_tests.py --performance            # Run performance tests
  python run_tests.py --specific tests/unit/test_fast_summarize.py
        """
    )
    
    # Test type options
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--chinese", action="store_true", help="Run Chinese language tests")
    parser.add_argument("--english", action="store_true", help="Run English language tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--visualization", action="store_true", help="Run visualization tests")
    
    # Test options
    parser.add_argument("--specific", type=str, help="Run specific test file or function")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🧪 LLM Text Summarization Tool - Test Runner")
    print(f"📁 Project directory: {project_dir}")
    
    # Check dependencies if requested
    if args.check_deps:
        return 0 if check_dependencies() else 1
    
    # Check dependencies before running tests
    if not check_dependencies():
        return 1
    
    # Determine which tests to run
    success = True
    
    if args.specific:
        success = run_specific_test(args.specific, args.verbose)
    elif args.unit:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    elif args.chinese:
        success = run_chinese_tests(args.verbose)
    elif args.english:
        success = run_english_tests(args.verbose)
    elif args.performance:
        success = run_performance_tests(args.verbose)
    elif args.visualization:
        success = run_visualization_tests(args.verbose)
    elif args.all:
        success = run_all_tests(args.verbose, args.coverage)
    else:
        # Default: run all tests
        success = run_all_tests(args.verbose, args.coverage)
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        print("📊 Check coverage report in htmlcov/ directory (if --coverage was used)")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())