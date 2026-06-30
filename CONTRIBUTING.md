# Contributing to ADE-FCM

We welcome contributions to ADE-FCM! This document outlines the guidelines for contributing.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs
1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include: Python version, OS, full traceback, minimal reproduction code

### Suggesting Features
1. Describe the feature and its motivation
2. Explain how it aligns with ADE-FCM's focus on fuzzy clustering
3. Provide example use cases

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Run linting: `flake8 novel_algorithm/`
6. Run type checking: `mypy novel_algorithm/`
7. Submit PR against the `main` branch

## Development Setup

```bash
git clone https://github.com/your-org/ade-fcm.git
cd ade-fcm
pip install -e ".[dev]"
pytest tests/
```

## Coding Standards

- **Style**: Black (line-length 110), isort (black profile)
- **Types**: Type hints required for all new functions
- **Docstrings**: NumPy-style docstrings for all public APIs
- **Tests**: Minimum 80% coverage for new code
- **Logging**: Use `loguru` logger, not `print()`

## Testing Guidelines

- Unit tests for each module in `tests/`
- Integration tests for cross-module functionality
- Benchmark tests for performance regression detection
- Run full suite: `pytest tests/ -v`

## Documentation

- Update relevant docs in `docs/` for API changes
- Update CHANGELOG.md for notable changes
- Add docstrings to all new public functions/classes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
