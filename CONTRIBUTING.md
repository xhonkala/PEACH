# Contributing to PEACH

Thank you for your interest in contributing to PEACH! We welcome contributions from the community and are excited to work with you.

##  Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/xhonkala/PEACH.git
   cd peach
   ```

2. **Create a virtual environment**
   ```bash
   conda create -n peach-dev python=3.9
   conda activate peach-dev
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

##  Development Process

### 1. Create an Issue First
Before starting work on a feature or bug fix, please create an issue to discuss:
- What you're planning to implement
- Why it's needed
- Your proposed approach

This helps avoid duplicate work and ensures alignment with project goals.

### 2. Branch Naming Convention
- Features: `feature/your-feature-name`
- Bug fixes: `fix/issue-description`
- Documentation: `docs/what-you-are-documenting`

### 3. Code Style Guidelines

#### Python Code
- Follow PEP 8 style guidelines
- Use type hints for function arguments and returns
- Maximum line length: 100 characters
- Use descriptive variable names

#### Documentation
- All public functions must have comprehensive docstrings
- Use NumPy-style docstrings:
  ```python
  def function_name(arg1: type, arg2: type) -> return_type:
      """
      Brief description of function.

      Parameters
      ----------
      arg1 : type
          Description of arg1
      arg2 : type
          Description of arg2

      Returns
      -------
      return_type
          Description of return value

      Examples
      --------
      >>> example_usage()
      expected_output
      """
  ```

### 4. Testing Requirements

- Write tests for all new functionality
- Maintain or improve code coverage (currently 96%)
- Tests go in `tests/` directory, mirroring `src/` structure
- Use pytest fixtures for common test data
- Include both unit tests and integration tests

Example test structure:
```python
def test_new_feature():
    """Test description."""
    # Arrange
    test_data = create_test_data()

    # Act
    result = your_function(test_data)

    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result.values, expected_values)
```

##  Architecture Guidelines

### Core Principles
1. **AnnData-centric**: All functions should work with AnnData objects
2. **Separation of concerns**: Keep `_core/` implementations separate from public API
3. **No phantom parameters**: Don't add parameters that aren't implemented
4. **Consistent naming**: Use exact names from `_core` in wrappers

### File Organization
```
src/peach/
 _core/           # Internal implementations
    models/      # Model definitions
    utils/       # Utility functions
    viz/         # Visualization internals
 pp/              # Preprocessing API
 tl/              # Tools API
 pl/              # Plotting API
```

### Adding New Features

1. **Implementation in `_core/`**
   - Add core functionality to appropriate module in `_core/`
   - Include comprehensive error handling
   - Add logging where appropriate

2. **Public API Wrapper**
   - Create simple wrapper in `pp/`, `tl/`, or `pl/`
   - Wrapper should only handle API translation, not logic
   - Update `__init__.py` to export new function

3. **Update Type System (Required)**
   - Add return type mapping to `types_index.py`
   - Add parameter schema to `tools_schema.py`
   - Add Pydantic model to `types.py` if needed
   - See [Type System Workflow](#type-system-workflow) below

4. **Documentation**
   - Add docstring with examples
   - Update relevant tutorial notebook
   - Add to API reference in README

5. **Testing**
   - Unit test for core implementation
   - Integration test for public API
   - Test with realistic data (not just synthetic)

## Type System Workflow

PEACH uses a structured type system for API consistency and agentic tool use. When adding or modifying functions, you **must** update these files in order:

### Ground Truth Files (in lookup order)

| File | Purpose | Size |
|------|---------|------|
| `src/peach/_core/types_index.py` | Function → return type mapping | ~270 lines |
| `src/peach/_core/tools_schema.py` | Function → input parameters | ~1000 lines |
| `src/peach/_core/types.py` | Full Pydantic type definitions | ~2500 lines |

### Step 1: Define Return Type in `types_index.py`

```python
# In types_index.py, add to RETURN_TYPE_MAP
RETURN_TYPE_MAP = {
    # ... existing entries ...
    "tl.your_new_function": ("YourReturnType", ["key1", "key2", "key3"]),
}
```

The tuple contains:
- Return type name (string matching a class in `types.py` or standard type)
- List of guaranteed keys/fields in the return value

### Step 2: Define Parameters in `tools_schema.py`

```python
# In tools_schema.py, add to TOOL_SCHEMAS
ToolSchema(
    name="tl.your_new_function",
    description="Brief description of what the function does",
    parameters=[
        ToolParameter(name="adata", type="AnnData", required=True,
                     description="Annotated data matrix"),
        ToolParameter(name="param1", type="int", required=True,
                     description="Description of param1"),
        ToolParameter(name="param2", type="float", required=False, default=0.5,
                     description="Description of param2"),
    ],
    returns="YourReturnType",
    returns_description="Description of return value",
)
```

### Step 3: Add Pydantic Model (if needed) in `types.py`

```python
# In types.py, add new result type if not using existing one
class YourReturnType(BaseModel):
    """Return type for tl.your_new_function."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key1: pd.DataFrame
    key2: np.ndarray
    key3: Optional[float] = None
```

### Validation Checklist

Before submitting a PR with new functions:

- [ ] Return type added to `types_index.py`
- [ ] Parameters added to `tools_schema.py`
- [ ] Pydantic model exists in `types.py` (or uses existing type)
- [ ] Function signature matches schema exactly
- [ ] Optional fields use `.get()` access pattern
- [ ] Run `python -c "from peach._core.types_index import get_return_type; print(get_return_type('tl.your_function'))"`

### Common Pitfalls

| Issue | Solution |
|-------|----------|
| KeyError on optional field | Use `.get('field_name')` not `['field_name']` |
| Schema mismatch | Ensure parameter names match exactly between schema and function |
| Missing from index | Function won't be discoverable by agents - always add to both files |

### Why This Matters

The type system enables:
1. **Agentic tool use** - AI agents can discover and compose PEACH functions
2. **Type validation** - Pydantic validates inputs/outputs at runtime
3. **Documentation generation** - Schemas auto-generate API docs
4. **IDE support** - Better autocomplete and type hints

##  Pull Request Process

### Before Submitting

1. **Run all tests**
   ```bash
   pytest tests/
   ```

2. **Check code style**
   ```bash
   flake8 src/
   black --check src/
   ```

3. **Update documentation**
   - Docstrings for new functions
   - Update README if adding features
   - Update tutorials if changing API

4. **Test tutorials**
   ```bash
   jupyter nbconvert --execute --to notebook docs/tutorials/*.ipynb
   ```

### PR Template

Your PR description should include:
```markdown
## Summary
Brief description of changes

## Motivation
Why are these changes needed?

## Changes Made
- List of specific changes
- Any breaking changes highlighted

## Testing
- How you tested the changes
- Any new tests added

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Tutorials still work
- [ ] No phantom parameters added
```

### Review Process

1. Automated checks run (tests, linting)
2. Code review by maintainer
3. Address feedback
4. Merge when approved

##  Reporting Bugs

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Code to reproduce
2. Expected behavior
3. Actual behavior

**Environment**
- Peach version:
- Python version:
- OS:
- PyTorch version:

**Additional Context**
Any other relevant information
```

##  Priority Areas for Contribution

### High Priority
- GPU optimization for large datasets
- Additional statistical tests
- Interactive visualization improvements
- Performance benchmarks

### Medium Priority
- Additional tutorials for specific use cases
- Integration with other single-cell tools
- Alternative model architectures
- Documentation improvements

### Good First Issues
- Improve error messages
- Add input validation
- Enhance test coverage
- Fix typos in documentation

## [STATS] Performance Guidelines

When adding features that process data:
- Consider memory usage for large datasets (>1M cells)
- Use sparse matrices where appropriate
- Add progress bars for long operations
- Profile your code for bottlenecks

##  Security

- Never commit sensitive data
- Don't include API keys or credentials
- Report security issues privately to maintainers

##  Communication

- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

##  Code of Conduct

Please note we have a code of conduct. By participating in this project, you agree to abide by its terms:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

##  Thank You!

Your contributions make PEACH better for everyone. We appreciate your time and effort in improving this project!

---

*For specific technical questions not covered here, please open a discussion or reach out to the maintainers.*