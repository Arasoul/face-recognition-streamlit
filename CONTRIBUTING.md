# Contributing to Face Recognition System

🎯 Thank you for your interest in contributing to the Face Recognition System! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/face-recognition-streamlit.git
   cd face-recognition-streamlit
   ```
3. **Set up the development environment** (see below)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA toolkit (optional, for GPU support)

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Required Development Tools
```bash
# Code formatting and linting
pip install black flake8 isort

# Testing
pip install pytest pytest-cov pytest-mock

# Documentation
pip install sphinx sphinx-rtd-theme
```

## 🔄 Contributing Process

### 1. Choose What to Work On
- Check [open issues](https://github.com/Arasoul/face-recognition-streamlit/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Propose new features in an issue first

### 2. Development Workflow
```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/your-feature

# 3. Make changes
# ... code changes ...

# 4. Test your changes
python -m pytest
python test_system.py

# 5. Format code
black .
isort .
flake8 .

# 6. Commit changes
git add .
git commit -m "feat: add your feature description"

# 7. Push and create PR
git push origin feature/your-feature
```

### 3. Pull Request Guidelines
- **Title**: Use conventional commits format
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for tests
  - `refactor:` for code refactoring

- **Description**: Include:
  - What changes were made
  - Why the changes were necessary
  - How to test the changes
  - Screenshots (for UI changes)

- **Checklist**:
  - [ ] Tests pass locally
  - [ ] Code is formatted with Black
  - [ ] Documentation is updated
  - [ ] CHANGELOG.md is updated
  - [ ] No merge conflicts

## 📝 Coding Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters

### Code Structure
```python
"""
Module docstring describing the purpose.
"""

import standard_library
import third_party_library

import local_module

# Constants
CONSTANT_VALUE = "value"

class ExampleClass:
    """Class docstring."""
    
    def __init__(self, param: str) -> None:
        """Initialize with parameters."""
        self.param = param
    
    def public_method(self, arg: int) -> str:
        """Public method with type hints and docstring."""
        return f"Result: {arg}"
    
    def _private_method(self) -> None:
        """Private method for internal use."""
        pass

def function_example(param: str, optional: bool = False) -> dict:
    """
    Function with type hints and docstring.
    
    Args:
        param: Description of parameter
        optional: Optional parameter with default
        
    Returns:
        Dictionary with results
        
    Raises:
        ValueError: When param is invalid
    """
    if not param:
        raise ValueError("param cannot be empty")
    
    return {"result": param, "optional": optional}
```

### Documentation Standards
- All public functions/classes need docstrings
- Use Google-style docstrings
- Include type hints for all parameters
- Add examples for complex functions

### Error Handling
```python
# Good: Specific exception handling
try:
    result = risky_operation()
except SpecificException as e:
    logging.error(f"Operation failed: {e}")
    raise ProcessingError(f"Failed to process: {e}") from e

# Bad: Bare except
try:
    result = risky_operation()
except:
    pass
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest test_system.py

# Run specific test
pytest test_system.py::TestFaceRecognitionSystem::test_model_initialization
```

### Writing Tests
```python
import pytest
from unittest.mock import Mock, patch

class TestYourFeature:
    """Test class for your feature."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_data = {"test": "data"}
    
    def test_feature_functionality(self):
        """Test the main functionality."""
        # Arrange
        input_data = "test_input"
        expected = "expected_output"
        
        # Act
        result = your_function(input_data)
        
        # Assert
        assert result == expected
    
    @patch('module.external_dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependencies."""
        mock_dependency.return_value = "mocked_result"
        
        result = function_using_dependency()
        
        assert result == "expected_with_mock"
        mock_dependency.assert_called_once()
```

### Test Coverage
- Aim for >90% test coverage
- Test both success and failure cases
- Include edge cases and error conditions
- Mock external dependencies

## 📚 Documentation

### Updating Documentation
- Update README.md for user-facing changes
- Update API.md for API changes
- Add docstrings for new functions/classes
- Update CHANGELOG.md

### Building Documentation
```bash
# Build Sphinx documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

## 🐛 Issue Reporting

### Bug Reports
Include:
- **Environment**: OS, Python version, package versions
- **Steps to reproduce**: Detailed steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traces
- **Screenshots**: If applicable

### Feature Requests
Include:
- **Problem description**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives**: Other possible solutions
- **Use cases**: Who would benefit?

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## 🎯 Areas for Contribution

### High Priority
- 🐛 Bug fixes
- 📚 Documentation improvements
- 🧪 Test coverage
- ⚡ Performance optimizations

### Medium Priority
- 🎨 UI/UX improvements
- 🔒 Security enhancements
- 🌐 Accessibility features
- 📱 Mobile responsiveness

### Future Features
- 🚀 REST API development
- 🔄 Real-time face tracking
- 🎭 Emotion detection
- 📊 Advanced analytics

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and discussion

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- GitHub releases

Thank you for contributing! 🎉
