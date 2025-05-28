<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Python Best Practices

## Project Structure

- Use src-layout with `src/your_package_name/`
- Place tests in `tests/` directory parallel to `src/`
- Keep configuration in `config/` or as environment variables
- Store requirements in `pyproject.toml`
- Place static files in `static/` directory
- Use `templates/` for Jinja2 templates
- Use `docs/` for documentation

## Code Style

- Follow ruff for linting
- Follow PEP 8 naming conventions:
    - snake_case for functions and variables
    - PascalCase for classes
    - UPPER_CASE for constants
- Use pydantic model, and all pydantic models should include `Field`, and `description` should be included.
- Maximum line length of 99 characters
- Use absolute imports over relative imports
    Example:

```python
from pydantic import BaseModel, Field


class User(BaseModel):
    """Example User model.

    Attributes:
        name (str): The name of the user
    """

    name: str = Field(..., description="The name of the user")


def foo(self, extra_input: str) -> str:
    """Example function.

    Args:
        extra_input (str): Extra input for the function

    Returns:
        str: Result of the function
    """
    return f"Hello, {self.name} and {extra_input}"
```

## Type Hints

- Use type hints for all function parameters and returns
- Use `TypeVar` for generic types
- Use `Protocol` for duck typing

## Testing

- Use pytest for testing
- Write tests for all routes
- Use pytest-cov for coverage
- Implement proper fixtures
- Use proper mocking with pytest-mock
- Test all error scenarios

## Performance

- Use proper caching with Flask-Caching
- Implement database query optimization
- Use proper connection pooling
- Implement proper pagination
- Use background tasks for heavy operations
- Monitor application performance

## Error Handling

- Create custom exception classes
- Use proper try-except blocks
- Implement proper logging
- Return proper error responses
- Handle edge cases properly
- Use proper error messages

## Documentation

- Use Google-style docstrings
- All documentation should be in English
- The most of the documentation should be in the code, but there are some exceptions:
    - `Installation` and `Project Background` is hard to include in the code, so it should be written in a markdown file under `docs/`
- Keep README.md updated
- Use proper inline comments for better mkdocs support
- Document environment setup

## Development Workflow

- Use virtual environments (venv)
- Implement pre-commit hooks
- Use proper Git workflow
- Follow semantic versioning and commit message conventions
- Use proper CI/CD practices

## Dependencies

- Use `uv` for dependency management
- Separate dev dependencies by adding `--dev` flag when adding dependencies
- Regularly update dependencies
