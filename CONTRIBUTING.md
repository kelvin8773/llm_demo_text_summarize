# Contributing to LLM Text Summarization Tool

Thank you for your interest in contributing to the LLM Text Summarization Tool! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report issues and bugs
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit code fixes or new features
- **Documentation**: Improve documentation and examples
- **Testing**: Add or improve tests
- **Performance**: Optimize code and models

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/llm_demo_text_summarize.git
   cd llm_demo_text_summarize
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Style

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting (configured in `pyproject.toml`)
- Maximum line length: 88 characters
- Use type hints where appropriate

### Code Formatting

```bash
# Format code with Black
black .

# Check code style
flake8 .

# Run type checking
mypy .
```

### Testing

- Write tests for new functionality
- Ensure all existing tests pass
- Aim for good test coverage

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=utils tests/
```

### Documentation

- Update README.md for significant changes
- Add docstrings to new functions and classes
- Include examples in docstrings
- Update API documentation if needed

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details**:
   - Python version
   - Operating system
   - Package versions
5. **Error messages** and stack traces
6. **Sample input** that causes the issue (if applicable)

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Streamlit: [e.g., 1.28.0]
- Transformers: [e.g., 4.30.0]

**Additional Context**
Add any other context about the problem here.
```

## ✨ Feature Requests

When requesting features, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**
5. **Additional context**

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
Describe how you think this feature should work.

**Alternatives**
Describe any alternative solutions you've considered.

**Additional Context**
Add any other context or screenshots about the feature request.
```

## 🔧 Code Contributions

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Submit a pull request**

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Code Review Process

- All pull requests require review
- Address feedback promptly
- Keep pull requests focused and small
- Update documentation and tests

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/
│   ├── test_fast_summarize.py
│   ├── test_enhance_summarize.py
│   ├── test_chinese_summarize.py
│   ├── test_insights.py
│   └── test_ingest.py
├── integration/
│   └── test_main.py
└── fixtures/
    └── sample_texts.py
```

### Writing Tests

```python
import pytest
from utils.fast_summarize import fast_summarize_text

def test_fast_summarize_basic():
    """Test basic summarization functionality."""
    text = "This is a sample text for testing summarization."
    result = fast_summarize_text(text, max_sentences=2)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) < len(text)

def test_fast_summarize_empty_input():
    """Test handling of empty input."""
    with pytest.raises(ValueError):
        fast_summarize_text("")
```

### Test Data

- Use small, representative test data
- Include edge cases (empty text, very long text, special characters)
- Avoid using copyrighted material
- Create reusable fixtures for common test data

## 📚 Documentation Guidelines

### Docstring Format

```python
def summarize_text(text: str, max_sentences: int = 5) -> str:
    """
    Summarize input text using transformer models.
    
    Args:
        text: Input text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Generated summary text
        
    Raises:
        ValueError: If input text is empty or invalid
        Exception: If summarization fails
        
    Example:
        >>> text = "This is a long article about..."
        >>> summary = summarize_text(text, max_sentences=3)
        >>> print(summary)
        "This is a summary..."
    """
```

### README Updates

- Update installation instructions for new dependencies
- Add new features to the features list
- Update usage examples
- Include new configuration options

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release notes prepared

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [Your Email] for direct contact

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to the LLM Text Summarization Tool! 🎉